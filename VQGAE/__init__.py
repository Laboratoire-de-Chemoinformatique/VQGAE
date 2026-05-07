"""VQGAE — Vector Quantised Graph AutoEncoder.

Public API. Heavy submodules (`preprocessing`, `cli`) are not re-exported here
to keep `import VQGAE` cheap; import them explicitly when needed.
"""

from .filters import (
    SMARTS_ALERTS,
    check_smarts_alerts,
    compute_qed_sas,
    mol_to_rdkit,
    pains_brenk_nih_filter,
)
from .inference import (
    decode_population,
    encode_smiles,
    vqgae_encode_dataset,
    vqgae_encode_mols,
)
from .models import VQGAE, OrderingNetwork
from .utils import (
    create_chem_graph,
    decode_molecules,
    extract_scaffold,
    filter_molecule,
    frag_counts_to_inds,
    frag_inds_to_counts,
    restore_order,
    tanimoto_distance_counts,
    tanimoto_kernel,
)

__all__ = [
    # post-processing — rdkit side (lazy import)
    "SMARTS_ALERTS",
    "VQGAE",
    "OrderingNetwork",
    "check_smarts_alerts",
    "compute_qed_sas",
    "create_chem_graph",
    "decode_molecules",
    "decode_population",
    # high-level
    "encode_smiles",
    # scaffold helpers
    "extract_scaffold",
    # post-processing — chython side
    "filter_molecule",
    "frag_counts_to_inds",
    "frag_inds_to_counts",
    "mol_to_rdkit",
    "pains_brenk_nih_filter",
    "restore_order",
    "tanimoto_distance_counts",
    # latent-space utilities
    "tanimoto_kernel",
    # batched encode entry points
    "vqgae_encode_dataset",
    "vqgae_encode_mols",
]
