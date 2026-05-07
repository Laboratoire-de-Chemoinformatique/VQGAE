# Latent representation reference

`VQGAE.encode(batch)` returns three tensors. Knowing which one to target is
the difference between a 30-second-to-prototype optimizer and a 30-line
research project.

```python
atoms_vectors, feature_vector, codebook_inds = vqgae_model.encode(batch)
```

| Tensor | Shape | Dtype | Source | What it is |
|--------|-------|-------|--------|------------|
| `codebook_inds` | `(B, max_atoms)` | `int64`, `-1` for padding | post-VQ | Discrete atom-fragment id per atom slot. |
| `atoms_vectors` | `(B, max_atoms, vector_dim)` | `float32` | post-VQ | Continuous atom-level vectors after quantization to the codebook. |
| `feature_vector` | `(B, vector_dim)` | `float32` | pre-VQ | Continuous molecule-level vector aggregated across atoms by `VQGraphAggregation`. |

For the published checkpoints, `max_atoms=51`, `vector_dim=512`, codebook
size 4096.

## Derived: `frag_counts` (the one you usually want)

```python
from VQGAE import frag_inds_to_counts
counts = frag_inds_to_counts(codebook_inds)   # (B, 4096) integer histogram
```

`frag_counts` is what every published GA + Optuna run searches over. Each
entry is the number of times codebook fragment `i` appears in the molecule.

Constraints:
- Non-negative integers
- `counts.sum(-1) ≤ max_atoms` (51)
- Decoder consumes this through `frag_counts_to_inds → restore_order → decode_molecules`
  (or just call `decode_population(counts, dec, onn)`)

## Which to optimize

| Optimizer | Operates on | Why |
|-----------|-------------|-----|
| **PyGAD GA** (single-objective) | `frag_counts` | Discrete, 4096-dim, integer genes. Fast. The recipe in tutorial 02. |
| **DEAP NSGA-II** (multi-objective) | `frag_counts` | Same space, multi-objective. Used for scaffold-constrained gen in tutorial 04. |
| **Optuna TPE** | `frag_counts` (sampled via `target_atoms` × `n_distinct` × `top_k_frags`) | Sample-efficient discrete BO. Tutorial 03. |
| **BoTorch / GP** (continuous BO) | `atoms_vectors` (B × 51 × 512 = 26112 dims) | Decode via `VQGAE.decode_from_atoms_vectors(...)`. Useful when your scoring function is differentiable or expensive. |
| **Continuous BO over molecule-level latent** | `feature_vector` (B × 512) | Currently the encoder produces it, but no decoder takes it directly. Would need a small `feature_vector → atoms_vectors` head trained on the codebook outputs. Open research extension. |

## Code skeletons

### Discrete BO (tutorial 03 path) — works today
```python
def objective(trial):
    counts = sample_candidate(trial)      # uses Optuna's suggest_int
    mols, valid, _ = decode_population(counts[None, :], dec, onn)
    if not valid[0]: return -1.0
    return score_molecule(mols[0])
```

### Continuous BO (uses `decode_from_atoms_vectors`) — also works today
```python
# inside your acquisition optimizer
import torch
B = candidate_batch.shape[0]
atoms_vectors = candidate_batch.view(B, 51, 512)
sizes = torch.full((B,), 51, dtype=torch.long)
atoms, bonds, sizes = vqgae_model.decode_from_atoms_vectors(atoms_vectors, sizes)
# convert (atoms, bonds, sizes) -> chython.MoleculeContainer via create_chem_graph
```
