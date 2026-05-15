import heapq

import click
import numpy as np
import torch
import yaml
from chython.containers import MoleculeContainer, QueryContainer
from chython.exceptions import InvalidAromaticRing
from chython.periodictable import AnyElement, ListElement
from chython.periodictable import O as O_elem
from scipy.optimize import linear_sum_assignment

# --- Architecture-defining constants -------------------------------------- #
# These match the published `tagirshin/VQGAE` checkpoint. Override at training
# time by setting `model.max_atoms` / `model.vq_embeddings` in the YAML config;
# inference helpers (`decode_population`, `vqgae_encode_dataset`, etc.) prefer
# the value carried on the model instance over these defaults so a model
# trained with different sizes works without per-call configuration.
DEFAULT_MAX_ATOMS: int = 51
DEFAULT_NUM_FRAGS: int = 4096


accepted_atoms = ("C", "N", "S", "O", "Se", "F", "Cl", "Br", "I", "B", "P", "Si")

atoms_types = (
    ("C", 0),
    ("S", 0),
    ("Se", 0),
    ("F", 0),
    ("Cl", 0),
    ("Br", 0),
    ("I", 0),
    ("B", 0),
    ("P", 0),
    ("Si", 0),
    ("O", 0),
    ("O", -1),
    ("N", 0),
    ("N", 1),
    ("N", -1),
)

training_config = {
    "seed_everything": 42,
    "trainer": {
        "logger": {
            "class_path": "pytorch_lightning.loggers.CSVLogger",
            "init_args": {
                "save_dir": "/home/.../logs",
                "name": "vqgae_default",
                "version": 1,
            },
        },
        "gradient_clip_val": 1.0,
        "gpus": 0,
        "max_epochs": 100,
        "log_every_n_steps": 1000,
        "precision": 16,
        "detect_anomaly": False,
    },
    "model": {
        "max_atoms": 51,
        "batch_size": 500,
        "num_conv_layers": 5,
        "vector_dim": 512,
        "num_mha_layers": 8,
        "num_agg_layers": 2,
        "num_heads_encoder": 16,
        "num_heads_decoder": 16,
        "dropout": 0.2,
        "vq_embeddings": 4096,
        "bias": True,
        "init_values": 0.0001,
        "lr": 0.0002,
        "task": "train",
        "shuffle_graph": False,
        "positional_bias": False,
        "reparam": False,
        "class_categories": [38, 29, 21, 25, 16, 13, 50, 13],
    },
    "data": {
        "path_train_predict": "/home/.../chembl_train.sdf",
        "path_val": "/home/.../chembl_val.sdf",
        "tmp_folder": "/home/.../tmp/",
        "tmp_name": "chembl",
        "num_workers": 0,
        "pin_memory": True,
        "drop_last": True,
        "properties_names": {
            "Heavy Atom Count": "class",
            "Hetero Atom Count": "class",
            "acceptorcount": "class",
            "donorcount": "class",
            "Chiral center count": "class",
            "Ring count": "class",
            "Hetero ring count": "class",
            "Rotatable bond count": "class",
            "Aromatic ring count": "class",
        },
    },
    "vqgae_lr_monitor": {
        "logging_interval": "epoch",
    },
    "vqgae_model_checkpoint": {
        "dirpath": "/home/.../weights",
        "filename": "${trainer.logger.init_args.name}",
        "monitor": "train_loss",
        "mode": "min",
    },
}


@click.command()
@click.option(
    "--task",
    required=True,
    help="For which task generate default config. Options are: train, encode, decode",
)
def vqgae_default_config(task):
    with open(f"default_vqgae_config_{task}.yaml", "w") as file:
        if task == "train" or task == "encode" or task == "decode":
            yaml.dump(training_config, file)
        else:
            raise ValueError(f"I don't know this task: {task}")


def create_chem_graph(atoms_sequence, connectivity_matrix, size) -> MoleculeContainer:
    """
    Create molecular graph or basis for HLG
    :param size: size of molecule
    :param atoms_sequence: sequence of atoms
    :param connectivity_matrix: adjacency matrix
    :return: molecular graph
    """
    molecule = MoleculeContainer()
    for n in range(size):
        atom = atoms_sequence[n]
        atomic_symbol, charge = atoms_types[int(atom)]  # defined on the line 9
        molecule.add_atom(atom=atomic_symbol, charge=charge)

    for i in range(len(molecule)):
        for j in range(i + 1, len(molecule)):
            if connectivity_matrix[i][j]:
                molecule.add_bond(i + 1, j + 1, int(connectivity_matrix[i][j]))

    return molecule


def beam_search(matrix, beam_width=5):
    # Initialize priority queue with an empty permutation and mean probability 0
    pq = [(-0.0, [])]
    best_perms = []

    for col in range(len(matrix[0])):
        new_pq = []

        # For each partial permutation in the priority queue
        for mean_prob, partial_perm in pq:
            for row in range(len(matrix)):
                if row not in partial_perm:
                    # Calculate new mean probability
                    new_mean_prob = (
                        (-mean_prob * len(partial_perm)) + matrix[row][col]
                    ) / (len(partial_perm) + 1)

                    # Add new partial permutation to the new priority queue
                    new_partial_perm = [*partial_perm, row]
                    heapq.heappush(new_pq, (-new_mean_prob, new_partial_perm))

                    # If the new partial permutation is complete, add it to the best_perms list
                    if len(new_partial_perm) == len(matrix[0]):
                        heapq.heappush(best_perms, (-new_mean_prob, new_partial_perm))

        # Keep only the top-k = beam_width partial permutations
        pq = heapq.nsmallest(beam_width, new_pq)

    # Get all complete permutations
    top_perms = heapq.nsmallest(beam_width, best_perms)
    top_perms = [(perm, -mean_prob) for mean_prob, perm in top_perms]
    return top_perms


def morgan(mol: MoleculeContainer) -> dict[int, float]:
    # Initialize atom values
    atom_vals = {}
    for idx, atom in mol._atoms.items():
        atom_num = (
            atom.atomic_number * 2
            - sum(int(b) for b in mol._bonds[idx].values())
            - atom.charge
        )
        if atom.in_ring:
            atom_num += 0.5
        atom_vals[idx] = atom_num

    limit = len(mol._atoms) - 1
    prev_count = 0
    stab_count = 0

    for _ in range(limit):
        # Update atom values based on neighbors
        new_vals = {}
        for idx in mol._atoms:
            new_val = atom_vals[idx] + sum(atom_vals[n] for n in mol._bonds[idx])
            new_vals[idx] = new_val

        atom_vals = new_vals

        # Check for uniqueness
        uniq_count = len(set(atom_vals.values()))
        if uniq_count == len(atom_vals):  # each atom now unique
            break
        elif uniq_count == prev_count:  # not changed. molecules like benzene
            if stab_count == 3:
                break
            stab_count += 1
        elif stab_count:  # changed unique atoms number. reset stability check.
            stab_count = 0

        prev_count = uniq_count

    return atom_vals


def extract_scaffold(mol: MoleculeContainer) -> MoleculeContainer:
    """Bemis-Murcko-style scaffold via chython's `skin_graph`.

    The scaffold is the union of:
      - all ring atoms of the molecule (chython's `skin_graph`),
      - any non-ring atom that is doubly-bonded to a ring atom (so e.g. exocyclic =O on a ketone is kept).

    Used as the substructure-match anchor in scaffold-constrained generation.
    Source: research/experiments/inverse_qsar/scaffold_selection.ipynb.
    """
    bm_framework = mol.skin_graph
    scaffold = MoleculeContainer()
    seen: set[int] = set()
    for atom in bm_framework:
        scaffold.add_atom(mol.atom(atom).atomic_symbol, atom)
        for neighbor in mol._bonds[atom]:
            if neighbor not in bm_framework and int(mol.bond(atom, neighbor)) > 1:
                scaffold.add_atom(mol.atom(neighbor).atomic_symbol, neighbor)
                scaffold.add_bond(atom, neighbor, mol.bond(atom, neighbor))
    for a, bs in bm_framework.items():
        seen.add(a)
        for b in bs:
            if b not in seen:
                scaffold.add_bond(a, b, mol.bond(a, b))
    scaffold.clean2d()
    return scaffold


def tanimoto_distance_counts(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pairwise Tanimoto **distance** = `1 - tanimoto_kernel(x, y)`.

    Convenience for diversity-driven fitness functions where you want a
    distance (higher = more dissimilar) rather than a similarity.
    """
    return 1 - tanimoto_kernel(x, y)


def tanimoto_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pairwise Tanimoto coefficient on count vectors.

    For two batches of count features X (N,D) and Y (M,D), returns the (N,M)
    matrix where T[i,j] = <x_i, y_j> / (|x_i|^2 + |y_j|^2 - <x_i, y_j>).
    Reduces to the Jaccard index for binary inputs. NaNs (when both rows are
    zero) are returned as 0.

    Used as the diversity term in inverse-QSAR fitness functions and as the
    similarity score for de-novo candidate filtering. Adapted from CIMtools.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    x_dot = x @ y.T
    x2 = (x**2).sum(axis=1)
    y2 = (y**2).sum(axis=1)
    with np.errstate(invalid="ignore"):
        result = x_dot / (np.add.outer(x2, y2) - x_dot)
    result[np.isnan(result)] = 0
    return result


def frag_inds_to_counts(raw_vectors, num_frags: int = DEFAULT_NUM_FRAGS):
    counts_vec = np.zeros((raw_vectors.shape[0], num_frags), dtype=np.uint8)
    for i in range(raw_vectors.shape[0]):
        for j in range(raw_vectors.shape[1]):
            cb_indx = raw_vectors[i, j]
            if cb_indx > -1:
                counts_vec[i, cb_indx] += 1
            else:
                break
    return counts_vec


def frag_counts_to_inds(frag_counts: np.ndarray, max_atoms: int = DEFAULT_MAX_ATOMS):
    frag_inds = -1 * torch.ones((frag_counts.shape[0], max_atoms), dtype=torch.int64)
    for mol_id in range(frag_counts.shape[0]):
        atom_counter = 0
        for frag_id in range(frag_counts.shape[1]):
            frag_count = frag_counts[mol_id, frag_id]
            if frag_count:
                for _ in range(frag_count):
                    frag_inds[mol_id, atom_counter] = frag_id
                    atom_counter += 1
                    if atom_counter == max_atoms:
                        break
                if atom_counter == max_atoms:
                    break
    return frag_inds


def find_best_permutation(perm_matrix: np.ndarray):
    num_rows, num_cols = np.shape(perm_matrix)
    if num_rows != num_cols:
        raise ValueError("The permutation matrix must be square.")

    original_cost_matrix = -np.array(perm_matrix)
    row_ind, col_ind = linear_sum_assignment(original_cost_matrix)
    score = -original_cost_matrix[row_ind, col_ind].mean()
    reorder_indices = np.argsort(col_ind)
    return reorder_indices, score


def restore_order(frag_inds, ordering_model):
    scores = []

    # Move input to the model's device — matches the canonical pattern in the
    # lab notebooks (.to(model.device)) so users can pass CPU tensors to a
    # cuda-loaded ordering_model without manual placement.
    device = next(ordering_model.parameters()).device
    frag_inds = frag_inds.to(device)

    canon_order_inds = -1 * torch.ones_like(frag_inds)

    mol_sizes = torch.where(frag_inds > -1, 1, 0).sum(-1)
    inputs_order_inds, _ = torch.sort(frag_inds, descending=True)

    with torch.no_grad():
        results = ordering_model([frag_inds]).cpu()

    for mol_i in range(frag_inds.shape[0]):
        mol_size = mol_sizes[mol_i]
        input_order_inds = inputs_order_inds[mol_i, :mol_size]
        result_mol = results[mol_i, :mol_size, :mol_size].numpy()
        top_solution, score = find_best_permutation(result_mol)
        canon_order_inds[mol_i, :mol_size] = input_order_inds[top_solution]
        scores.append(score)

    return canon_order_inds, scores


# bad groups filtration rules

small_ring = QueryContainer("")
small_ring.add_atom(AnyElement())
small_ring.add_atom(AnyElement())
small_ring.add_atom(AnyElement())
small_ring.add_atom(AnyElement())
small_ring.add_bond(1, 2, [2, 4])
small_ring.add_bond(3, 4, [2, 4])
small_ring.add_bond(1, 3, [1, 2, 3, 4])
small_ring.add_bond(2, 4, [1, 2, 3, 4])

bad_acylator = QueryContainer("")
bad_acylator.add_atom(AnyElement())
bad_acylator.add_atom(ListElement(["F", "Cl", "Br", "I"]))
bad_acylator.add_atom(ListElement(["O", "N"]))
bad_acylator.add_bond(1, 2, 1)
bad_acylator.add_bond(1, 3, 2)

bad_heterotriplet = QueryContainer("")
bad_heterotriplet.add_atom(ListElement(["B", "N", "O"]))
bad_heterotriplet.add_atom(ListElement(["B", "N", "O"]))
bad_heterotriplet.add_atom(ListElement(["B", "N", "O"]))
bad_heterotriplet.add_bond(1, 2, [2, 3, 4])
bad_heterotriplet.add_bond(
    1,
    3,
    [
        1,
        2,
    ],
)

bad_multiring = QueryContainer("")
for _ in range(7):
    bad_multiring.add_atom(AnyElement())
bad_multiring.add_bond(1, 2, 4)
bad_multiring.add_bond(2, 3, 4)
bad_multiring.add_bond(3, 4, 4)
bad_multiring.add_bond(4, 5, 4)
bad_multiring.add_bond(5, 6, 4)
bad_multiring.add_bond(6, 1, 4)
bad_multiring.add_bond(1, 7, 1)
bad_multiring.add_bond(3, 7, 1)

bad_multiring_2 = QueryContainer("")
for _ in range(5):
    bad_multiring_2.add_atom("C")
for _ in range(2):
    bad_multiring_2.add_atom(AnyElement())
bad_multiring_2.add_bond(1, 2, 4)
bad_multiring_2.add_bond(2, 3, 4)
bad_multiring_2.add_bond(3, 4, 4)
bad_multiring_2.add_bond(4, 5, 4)
bad_multiring_2.add_bond(5, 1, 4)
bad_multiring_2.add_bond(1, 6, 4)
bad_multiring_2.add_bond(3, 7, 4)

allene = QueryContainer("")
allene.add_atom("C")
allene.add_atom("A")
allene.add_atom("A")
allene.add_bond(1, 2, 2)
allene.add_bond(1, 3, 2)

peroxide_charge = QueryContainer("")
peroxide_charge.add_atom(O_elem(charge=-1))
peroxide_charge.add_atom("O")
peroxide_charge.add_bond(1, 2, 1)

peroxide = QueryContainer("")
peroxide.add_atom("O")
peroxide.add_atom("O")
peroxide.add_bond(1, 2, 1)

bad_groups = [
    small_ring,
    bad_acylator,
    bad_heterotriplet,
    bad_multiring,
    bad_multiring_2,
    allene,
    peroxide_charge,
    peroxide,
]


def filter_molecule(molecule: MoleculeContainer):
    # filtering out molecules with the bad group
    for group in bad_groups:
        if group < molecule:
            return False

    # filtering out macrocycles and wrong microcycles
    for ring in molecule.sssr:
        if len(ring) > 8:
            return False
        elif len(ring) == 3:
            for n_atom in ring:
                atom = molecule.atom(n_atom)
                if atom.hybridization != 1 or atom.atomic_symbol not in ["C", "O"]:
                    return False
    return True


def decode_molecules(ordered_frag_inds, vqgae_model, clean_2d=True):
    decoded_molecules = []
    validity = []
    # Move to model device so a CPU input can hit a cuda-loaded model.
    device = next(vqgae_model.parameters()).device
    ordered_frag_inds = ordered_frag_inds.to(device)
    prediction = vqgae_model([ordered_frag_inds])
    # the model returns (atoms, bonds, sizes) — move to CPU for numpy-side work
    prediction = tuple(t.cpu() for t in prediction)

    for mol_i in range(prediction[0].shape[0]):
        molecule = create_chem_graph(
            prediction[0][mol_i],
            prediction[1][mol_i],
            int(prediction[2][mol_i]),
        )
        valid = False
        if len(molecule) > 2:
            if clean_2d:
                molecule.clean2d()
            if molecule.connected_components_count == 1:
                if not molecule.check_valence():
                    try:
                        molecule.thiele()
                        valid = filter_molecule(molecule)
                    except InvalidAromaticRing:
                        valid = False
        decoded_molecules.append(molecule)
        validity.append(valid)
    return decoded_molecules, validity


def decoded_mol(atoms, bonds, size: int, canonicalize: bool = False):
    """chython MoleculeContainer from a decoded (atoms, bonds, size) triple,
    or None if invalid (disconnected, bad valence, unaromatisable).

    chython's ``MoleculeContainer.__eq__`` resolves to ``str(self) == str(other)``,
    so equality across atom orderings (Stage 2 / full reconstruction, where the
    ONN may reshuffle slots relative to GT) needs ``canonicalize=True``. Stage 1
    keeps GT ordering by construction, so canonicalisation is unnecessary there.
    """
    if size < 1:
        return None
    mol = create_chem_graph(atoms, bonds, int(size))
    if len(mol) < 2 or mol.connected_components_count != 1:
        return None
    if mol.check_valence():
        return None
    try:
        mol.thiele()
    except InvalidAromaticRing:
        return None
    if canonicalize:
        try:
            mol.canonicalize()
        except Exception:
            return None
    return mol
