import numpy as np
import torch
from chython.exceptions import InvalidAromaticRing
from torch_geometric.data import Batch
from torch_geometric.loader.dataloader import DataLoader as PYGDataLoader
from tqdm import tqdm

from VQGAE.preprocessing import MolDataset, preprocess_molecule
from VQGAE.utils import (
    DEFAULT_MAX_ATOMS,
    DEFAULT_NUM_FRAGS,
    create_chem_graph,
    filter_molecule,
    find_best_permutation,
)
from VQGAE.utils import (
    decode_molecules as _decode_molecules_torch,
)

# torch-tensor versions used by decode_population (this module's own
# frag_counts_to_inds / restore_order / decode_molecules are numpy-based and
# would shadow these names without the explicit aliases).
from VQGAE.utils import (
    frag_counts_to_inds as _fc2i_torch,
)
from VQGAE.utils import (
    restore_order as _restore_order_torch,
)


def _model_max_atoms(model, fallback: int = DEFAULT_MAX_ATOMS) -> int:
    """Read ``max_atoms`` from a Lightning module if available; else fallback.

    Lets the inference helpers stay correct when a user trains the model with
    a non-default ``max_atoms`` and then calls e.g. ``decode_population``
    without re-passing the same value.
    """
    return int(getattr(model, "max_atoms", fallback))


def vqgae_encode_dataset(
    data_file,
    vqgae_model,
    batch_size: int = 10,
    max_atoms: int | None = None,
):
    """Encode every molecule in an SDF, returning per-mol codebook indices.

    :param max_atoms: graph size cap; if None, derived from
        ``vqgae_model.max_atoms`` (so the value matches what the model was
        trained with). Pass an explicit int to override.
    """
    if max_atoms is None:
        max_atoms = _model_max_atoms(vqgae_model)
    data = MolDataset(max_atoms=max_atoms, molecules_file=data_file)
    loader = PYGDataLoader(data, batch_size=batch_size)
    results = []
    for batch in tqdm(loader):
        _, _, codebook_inds = vqgae_model(batch)
        results.extend(codebook_inds.tolist())
    results = np.array(results)
    return results


def encode_smiles(smiles_list, vqgae_encoder, batch_size: int = 10):
    """Parse SMILES strings with chython, encode, return (codebook_inds, frag_counts).

    Convenience wrapper for the common case where the user starts from SMILES.
    Internally it does:
      smiles -> chython.MoleculeContainer (canonicalised)
             -> vqgae_encode_mols(...)               # codebook_inds (N, max_atoms)
             -> frag_inds_to_counts(...)             # frag_counts (N, num_frags)

    :param smiles_list: list[str] of SMILES.
    :param vqgae_encoder: VQGAE loaded with task='encode'.
    :param batch_size: forward-pass batch size.
    :return: (codebook_inds, frag_counts) — both numpy arrays.
    """
    from chython import smiles as _parse_smiles  # lazy import to keep package light

    mols = []
    for s in smiles_list:
        m = _parse_smiles(s)
        m.canonicalize()
        mols.append(m)
    codebook_inds = vqgae_encode_mols(mols, vqgae_encoder, batch_size=batch_size)
    counts = frag_inds_to_counts(codebook_inds)
    return codebook_inds, counts


def decode_population(
    counts: np.ndarray,
    vqgae_decoder,
    ordering_model,
    batch_size: int = 100,
    max_atoms: int | None = None,
    clean_2d: bool = False,
):
    """Decode a population of fragment-count vectors back to molecules.

    Wraps the canonical three-step decode pipeline used in every published
    notebook:
        counts -> frag_counts_to_inds -> restore_order(ordering_model)
               -> decode_molecules(vqgae_decoder)

    :param counts: (N, num_frags) integer histogram, sum per row <= max_atoms.
    :param vqgae_decoder: VQGAE loaded with task='decode'.
    :param ordering_model: OrderingNetwork (used to pick a canonical permutation).
    :param batch_size: forward-pass batch size for decoding.
    :param max_atoms: graph size cap; if None, derived from
        ``vqgae_decoder.max_atoms`` so the value matches the model's
        training-time configuration. Pass an int to override.
    :param clean_2d: if True, run molecule.clean2d() during validity check.
    :return: tuple (molecules, validity, ordering_scores).
        - molecules: list[chython.MoleculeContainer]
        - validity: list[bool], True iff the decoded molecule passes filter_molecule
        - ordering_scores: list[float], mean per-atom permutation confidence
    """
    if max_atoms is None:
        max_atoms = _model_max_atoms(vqgae_decoder)
    counts = np.asarray(counts)
    if counts.ndim == 1:
        counts = counts[None, :]

    all_mols, all_valid, all_scores = [], [], []
    for start in tqdm(range(0, counts.shape[0], batch_size), desc="decode"):
        chunk = counts[start : start + batch_size]
        inds = _fc2i_torch(chunk, max_atoms=max_atoms)
        canon, ord_scores = _restore_order_torch(inds, ordering_model)
        mols, validity = _decode_molecules_torch(
            canon, vqgae_decoder, clean_2d=clean_2d
        )
        all_mols.extend(mols)
        all_valid.extend(bool(v) for v in validity)
        all_scores.extend(ord_scores)
    return all_mols, all_valid, all_scores


def vqgae_encode_mols(molecules: list, vqgae_model, batch_size=10):
    def process_batch(batch, results, pbar):
        inp_data = Batch.from_data_list(batch)
        inp_data = inp_data.to(vqgae_model.device)
        _, _, codebook_inds = vqgae_model.encode(inp_data)
        # The model's `to_dense_batch` pads to vqgae_model.batch_size — slice
        # off the padding rows so the output length equals the input length.
        results.extend(codebook_inds[: len(batch)].tolist())
        pbar.update(1)

    results = []
    batch = []
    last_batch = 1 if len(molecules) % batch_size else 0
    with tqdm(total=len(molecules) // batch_size + last_batch) as pbar:
        for molecule in molecules:
            batch.append(preprocess_molecule(molecule))
            if len(batch) == batch_size:
                process_batch(batch, results, pbar)
                batch = []
        else:
            if batch:
                process_batch(batch, results, pbar)
                batch = []
    results = np.array(results)
    return results


def frag_inds_to_counts(raw_vectors, num_frags: int = DEFAULT_NUM_FRAGS):
    counts_vec = np.zeros((raw_vectors.shape[0], num_frags), dtype=np.uint8)

    for i in range(raw_vectors.shape[0]):
        unique, counts = np.unique(
            raw_vectors[i, raw_vectors[i] > -1], return_counts=True
        )
        counts_vec[i, unique] = counts

    return counts_vec


def frag_counts_to_inds(frag_counts: np.ndarray, max_atoms: int = DEFAULT_MAX_ATOMS):
    num_molecules, num_frags = frag_counts.shape
    frag_inds = -1 * np.ones((num_molecules, max_atoms), dtype=np.int64)

    for mol_id in range(num_molecules):
        # Get the fragment indices and counts for the current molecule
        frags = np.repeat(np.arange(num_frags), frag_counts[mol_id])

        # Limit to max_atoms and update frag_inds
        if len(frags) > max_atoms:
            frags = frags[:max_atoms]
        frag_inds[mol_id, : len(frags)] = frags

    return frag_inds


def vqgae_restore_order(frag_inds: np.ndarray, ordering_model):
    scores = []
    canon_order_inds = -1 * np.ones_like(frag_inds)
    mol_sizes = np.where(frag_inds > -1, 1, 0).sum(-1)
    inp_sorted_indices = np.argsort(-frag_inds, axis=1)
    inp_sorted_frag_inds = np.take_along_axis(frag_inds, inp_sorted_indices, axis=1)

    with torch.no_grad():
        results = ordering_model(
            [torch.from_numpy(frag_inds).to(ordering_model.device)]
        )
        results = results.cpu().numpy()

    for mol_i in range(frag_inds.shape[0]):
        mol_size = mol_sizes[mol_i].item()
        if mol_size > 2:
            input_order_inds = inp_sorted_frag_inds[mol_i, :mol_size]
            result_mol = results[mol_i, :mol_size, :mol_size]
            top_solution, score = find_best_permutation(result_mol)
            canon_order_inds[mol_i, :mol_size] = input_order_inds[top_solution]
            scores.append(score)
        else:
            canon_order_inds[mol_i, :mol_size] = inp_sorted_frag_inds[mol_i, :mol_size]
            scores.append(0)
    return canon_order_inds, scores


def vqgae_decode(ordered_frag_inds, vqgae_model, clean_2d=True):
    decoded_molecules = []
    validity = []
    atoms, bonds, mol_sizes = vqgae_model(
        [torch.from_numpy(ordered_frag_inds).to(vqgae_model.device)]
    )
    atoms = atoms.cpu().numpy()
    bonds = bonds.cpu().numpy()
    mol_sizes = mol_sizes.cpu().numpy()
    for mol_i in range(atoms.shape[0]):
        molecule = create_chem_graph(
            atoms[mol_i],
            bonds[mol_i],
            int(mol_sizes[mol_i]),
        )
        valid = False
        if len(molecule) > 2:
            if molecule.connected_components_count == 1:
                if not molecule.check_valence():
                    try:
                        molecule.thiele()
                        valid = filter_molecule(molecule)
                    except InvalidAromaticRing:
                        valid = False
                    if clean_2d and valid:
                        try:
                            molecule.clean2d()
                        except:
                            valid = False
        decoded_molecules.append(molecule)
        validity.append(valid)
    return decoded_molecules, validity
