"""Preprocessing pipelines for VQGAE.

Two parallel front-ends sit behind a single import surface:

- :mod:`VQGAE.preprocessing.chython` — canonical chython path. Strict valences,
  deep query / scaffold support. Used by the inference + analysis stack.
- :mod:`VQGAE.preprocessing.rdkit` — fast rdkit path. ~10x faster SMILES parse,
  same atom ordering (Morgan-hash BFS) as the chython path. Used by the
  high-throughput sharded preprocessor for SMILES / SDF training corpora.

Backwards compatibility: every name that used to live in
``VQGAE.preprocessing`` (when it was a flat module) is re-exported from this
``__init__`` so ``from VQGAE.preprocessing import VQGAEData`` etc. keeps
working unchanged.
"""

from .chython import (
    MolDataset,
    VQGAEData,
    VQGAEVectors,
    atom_to_vector,
    bfs_remap,
    bonds_to_vector,
    calc_atoms_info,
    graph_to_atoms_true_vector,
    graph_to_atoms_vectors,
    graph_to_bond_matrix,
    mendel_info,
    preprocess_molecule,
    preprocess_molecules,
)
from .rdkit import preprocess_rdkit
from .shards import (
    ShardedVQGAEData,
    ShardedVQGAEDataset,
    preprocess_to_shards,
    write_shard,
)
from .smi import (
    DEFAULT_PROP_BINS,
    clip_props_to_bins,
    compute_rdkit_props,
    iter_smiles,
)

__all__ = [
    "DEFAULT_PROP_BINS",
    "MolDataset",
    # sharded preprocessing pipeline
    "ShardedVQGAEData",
    "ShardedVQGAEDataset",
    # chython path (canonical)
    "VQGAEData",
    "VQGAEVectors",
    "atom_to_vector",
    "bfs_remap",
    "bonds_to_vector",
    "calc_atoms_info",
    "clip_props_to_bins",
    "compute_rdkit_props",
    "graph_to_atoms_true_vector",
    "graph_to_atoms_vectors",
    "graph_to_bond_matrix",
    # SMILES / property helpers
    "iter_smiles",
    "mendel_info",
    "preprocess_molecule",
    "preprocess_molecules",
    # rdkit path (high-throughput)
    "preprocess_rdkit",
    "preprocess_to_shards",
    "write_shard",
]
