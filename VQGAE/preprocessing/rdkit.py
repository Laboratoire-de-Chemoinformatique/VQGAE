"""rdkit-backed alternative to ``preprocessing.preprocess_molecule``.

Same tensor contract (x: int8 (max_atoms, 11), atoms_types: int8 (max_atoms,),
edge_index: int (2, 2E), edge_attr: int (2E,), mol_size: int) but produced
from an ``rdkit.Chem.Mol`` instead of a chython ``MoleculeContainer``.

Why a parallel path:
    chython's pure-Python SMILES parser is ~10x slower than rdkit's C++ one.
    For 10 M+ training sets the read becomes the bottleneck, not the model.
    The chython path is kept as the canonical inference / generation path
    (extract_scaffold, filter_molecule, etc. live there); rdkit is used here
    purely as a fast preprocessing front-end.

Reproducibility note: ``_bfs_remap_rdkit`` re-implements the chython
``utils.bfs_remap`` exactly (graph centre + Morgan-hash-sorted BFS), so a
molecule preprocessed via this path produces the same atom ordering — and
therefore the same codebook indices when run through the encoder — as the
chython path. We assert this on a small sample in the test suite.
"""

from __future__ import annotations

from collections import deque

import networkx as nx
import numpy as np
from rdkit import Chem

from ..utils import DEFAULT_MAX_ATOMS, atoms_types
from .chython import mendel_info

# atoms_types is the (symbol, charge) -> int index table from utils.py;
# build the reverse lookup once.
_ATOM_TYPE_INDEX: dict[tuple[str, int], int] = {
    pair: i for i, pair in enumerate(atoms_types)
}


def _morgan_rdkit(rdkit_mol: Chem.Mol) -> dict[int, float]:
    """Port of ``VQGAE.utils.morgan`` to rdkit.

    Returns a dict ``{atom_idx_0based: hash}`` used to break ties when
    sorting neighbours during ``_bfs_remap_rdkit``.

    Algorithm: seed each atom with ``atomic_number*2 - bond_order_sum -
    charge`` (+0.5 if in a ring); iteratively refine by adding the sum
    of neighbour values; stop when all atoms become unique or when the
    number of unique values stops changing for 3 iterations.
    """
    atom_vals: dict[int, float] = {}
    for atom in rdkit_mol.GetAtoms():
        idx = atom.GetIdx()
        bond_sum = sum(int(b.GetBondTypeAsDouble()) for b in atom.GetBonds())
        val = atom.GetAtomicNum() * 2 - bond_sum - atom.GetFormalCharge()
        if atom.IsInRing():
            val += 0.5
        atom_vals[idx] = val

    n = rdkit_mol.GetNumAtoms()
    prev_count = 0
    stab_count = 0
    for _ in range(n - 1):
        new_vals: dict[int, float] = {}
        for atom in rdkit_mol.GetAtoms():
            idx = atom.GetIdx()
            new_vals[idx] = atom_vals[idx] + sum(
                atom_vals[nb.GetIdx()] for nb in atom.GetNeighbors()
            )
        atom_vals = new_vals

        uniq_count = len(set(atom_vals.values()))
        if uniq_count == n:
            break
        if uniq_count == prev_count:
            if stab_count == 3:
                break
            stab_count += 1
        elif stab_count:
            stab_count = 0
        prev_count = uniq_count
    return atom_vals


def _bfs_remap_rdkit(rdkit_mol: Chem.Mol, hashes: dict[int, float]) -> list[int]:
    """Return a permutation: new_position -> original 0-based atom idx.

    Walks the bond graph BFS from one of nx's ``center`` nodes, sorting
    neighbours at each step by their Morgan hash so the order is
    deterministic across kekulisation and explicit-H differences.
    """
    g = nx.Graph()
    g.add_nodes_from(atom.GetIdx() for atom in rdkit_mol.GetAtoms())
    for bond in rdkit_mol.GetBonds():
        g.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    # `center` returns one or more nodes equidistant from the rest. The
    # chython path picks the first; we do the same for parity.
    central = nx.center(g)[0]
    visited: list[int] = [central]
    seen: set[int] = {central}
    queue: deque[int] = deque([central])
    while len(visited) != g.number_of_nodes():
        cur = queue.popleft()
        nbrs = sorted(g.neighbors(cur), key=lambda i: hashes[i])
        for nb in nbrs:
            if nb not in seen:
                visited.append(nb)
                seen.add(nb)
                queue.append(nb)
    return visited


def preprocess_rdkit(
    rdkit_mol: Chem.Mol,
    *,
    max_atoms: int = DEFAULT_MAX_ATOMS,
    class_props: list[int] | None = None,
) -> dict | None:
    """Build the per-molecule tensor dict expected by the shard writer.

    :param rdkit_mol: an ``rdkit.Chem.Mol``. If parsing failed upstream and
        ``None`` is passed, returns ``None`` to make filtering trivial.
    :param max_atoms: must match the model's ``max_atoms`` (51 for the
        published checkpoint). Molecules with more heavy atoms are dropped.
    :param class_props: optional list of integer class labels for the
        multi-target classifier head. Length must be consistent across
        all molecules in a shard.
    :return: dict with keys ``x, atoms_types, mol_size, edge_index,
        edge_attr`` (and ``class_prop`` if labels were supplied), or
        ``None`` if the molecule should be skipped (oversized, hypervalent,
        unknown atom type, kekulisation failure).
    """
    if rdkit_mol is None:
        return None

    # Drop explicit Hs — the chython path also operates on heavy atoms.
    if any(a.GetAtomicNum() == 1 for a in rdkit_mol.GetAtoms()):
        rdkit_mol = Chem.RemoveHs(rdkit_mol)

    n_atoms = rdkit_mol.GetNumHeavyAtoms()
    if n_atoms == 0 or n_atoms >= max_atoms:
        return None

    # Bond orders need to be 1/2/3 for `bonds_to_vector` to fit in (single,
    # double, triple) bins. Aromatic (1.5) is not allowed.
    try:
        Chem.Kekulize(rdkit_mol, clearAromaticFlags=True)
    except Exception:
        return None

    # Canonical atom ordering — same algorithm as chython's bfs_remap so
    # the model sees the same input for a given molecule regardless of
    # which front-end we used.
    try:
        hashes = _morgan_rdkit(rdkit_mol)
        order = _bfs_remap_rdkit(rdkit_mol, hashes)
    except (nx.NetworkXError, nx.NetworkXPointlessConcept):
        # Disconnected molecule — nx center is undefined.
        return None
    new_idx = {old: new for new, old in enumerate(order)}

    x = np.zeros((max_atoms, 11), dtype=np.int8)
    atoms_types_vec = np.full(max_atoms, -1, dtype=np.int8)
    bond_counts = np.zeros((n_atoms, 3), dtype=np.int8)

    for old_idx, new_i in new_idx.items():
        atom = rdkit_mol.GetAtomWithIdx(old_idx)
        sym = atom.GetSymbol()
        info = mendel_info.get(sym)
        if info is None:
            return None  # atom outside the 12-element vocabulary
        period, group, shell, electrons = info

        x[new_i, 0] = atom.GetAtomicNum()
        x[new_i, 1] = period
        x[new_i, 2] = group
        x[new_i, 3] = electrons + atom.GetFormalCharge()
        x[new_i, 4] = shell
        x[new_i, 5] = atom.GetTotalNumHs()
        x[new_i, 6] = int(atom.IsInRing())
        x[new_i, 7] = atom.GetDegree()

        type_key = (sym, atom.GetFormalCharge())
        type_idx = _ATOM_TYPE_INDEX.get(type_key)
        if type_idx is None:
            return None  # (atom, charge) outside the 15-element vocabulary
        atoms_types_vec[new_i] = type_idx

    # Bonds: store both directions (the chython path runs ToUndirected()
    # afterwards, which doubles edges; we do that explicitly here so the
    # shard format is already-undirected and PyG sees no asymmetry).
    src: list[int] = []
    dst: list[int] = []
    orders: list[int] = []
    for bond in rdkit_mol.GetBonds():
        order_int = int(bond.GetBondTypeAsDouble())
        if order_int not in (1, 2, 3):
            return None
        a = new_idx[bond.GetBeginAtomIdx()]
        b = new_idx[bond.GetEndAtomIdx()]
        src.extend([a, b])
        dst.extend([b, a])
        orders.extend([order_int, order_int])
        bond_counts[a, order_int - 1] += 1
        bond_counts[b, order_int - 1] += 1

    if not src:
        return None  # single atom without bonds — model can't use this

    for i in range(n_atoms):
        x[i, 8:11] = bond_counts[i]

    out: dict = {
        "x": x,
        "atoms_types": atoms_types_vec,
        "mol_size": np.int8(n_atoms),
        "edge_index": np.array([src, dst], dtype=np.int32),
        "edge_attr": np.array(orders, dtype=np.int8),
    }
    if class_props is not None:
        out["class_prop"] = np.asarray(class_props, dtype=np.int8)
    return out


__all__ = ["preprocess_rdkit"]
