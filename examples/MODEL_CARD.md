---
license: mit
library_name: pytorch
tags:
  - chemistry
  - molecular-generation
  - graph-neural-network
  - autoencoder
  - vector-quantization
  - inverse-qsar
pipeline_tag: graph-ml
---

# VQGAE — Vector Quantised Graph AutoEncoder

VQGAE encodes molecules into an **order-independent** discrete latent: a histogram over a 4096-entry codebook of atom-environment fragments, capped at 51 atoms. The reverse direction is a permutation-aware decoder that reconstructs the molecular graph (atoms + bonds) from a chosen fragment ordering. This makes the latent space compatible with combinatorial optimizers (GA, TPE, simulated annealing) without the canonical-SMILES-ordering problem.

- **Paper:** Akhmetshin, Lin, Madzhidov, Varnek (2023). *Construction of order-independent molecular fragments space with vector quantised graph autoencoder.* ChemRxiv. [doi:10.26434/chemrxiv-2023-5zmvw](https://doi.org/10.26434/chemrxiv-2023-5zmvw)
- **Code:** https://github.com/Laboratoire-de-Chemoinformatique/VQGAE
- **Demo:** https://huggingface.co/spaces/tagirshin/VQGAE (Tubulin inverse-QSAR with PyGAD GA)

## Files

| File | Purpose |
|---|---|
| `vqgae.ckpt` | VQGAE encoder + vector quantizer + decoder. Loadable as `VQGAE.load_from_checkpoint(..., task='encode'\|'decode')`. |
| `ordering_network.ckpt` | Auxiliary network that scores atom orderings; used to pick a canonical permutation before decoding. Loadable as `OrderingNetwork.load_from_checkpoint(...)`. |

Both files are PyTorch Lightning checkpoints (`.ckpt`).

## Training data

Trained on a filtered ChEMBL subset (≤ 51 heavy atoms). Atom vocabulary (15 entries):
`(C,0), (S,0), (Se,0), (F,0), (Cl,0), (Br,0), (I,0), (B,0), (P,0), (Si,0), (O,0), (O,−1), (N,0), (N,+1), (N,−1)`.
Multi-target classification heads provided weak supervision over: heavy-atom count, hetero-atom count, H-bond acceptor/donor counts, chiral centers, ring counts, hetero-ring counts, rotatable bonds, aromatic-ring count.

## Quickstart

```bash
pip install "vqgae[cpu,examples] @ git+https://github.com/Laboratoire-de-Chemoinformatique/VQGAE.git"
```

```python
from huggingface_hub import hf_hub_download
from chython import smiles
from VQGAE.models import VQGAE, OrderingNetwork
from VQGAE.utils import frag_counts_to_inds, restore_order, decode_molecules
from VQGAE.inference import vqgae_encode_mols, frag_inds_to_counts

# Download checkpoints from this repo
vqgae_path = hf_hub_download("tagirshin/VQGAE", "vqgae.ckpt")
onn_path   = hf_hub_download("tagirshin/VQGAE", "ordering_network.ckpt")

# Load (separate task heads share weights)
enc = VQGAE.load_from_checkpoint(vqgae_path, task="encode", batch_size=4, map_location="cpu").eval()
dec = VQGAE.load_from_checkpoint(vqgae_path, task="decode", batch_size=4, map_location="cpu").eval()
onn = OrderingNetwork.load_from_checkpoint(onn_path, batch_size=4, map_location="cpu").eval()

# Round-trip a few molecules
mols = [smiles(s) for s in ["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]]
for m in mols: m.canonicalize()

codebook_inds = vqgae_encode_mols(mols, enc, batch_size=len(mols))     # (N, 51) int
counts        = frag_inds_to_counts(codebook_inds, num_frags=4096)     # (N, 4096) histogram
frag_inds     = frag_counts_to_inds(counts, max_atoms=51)              # back to (N, 51)
ordered, _    = restore_order(frag_inds, onn)                          # canonical order
decoded, ok   = decode_molecules(ordered, dec, clean_2d=False)
```

Full end-to-end script: [`examples/smoke_test.py`](https://github.com/Laboratoire-de-Chemoinformatique/VQGAE/blob/main/examples/smoke_test.py).

## Latent representation — what an optimizer can search over

| Tensor | Shape | Type | Where it comes from |
|---|---|---|---|
| `codebook_inds` | (B, 51) | int64, −1 = pad | `VQGAE.encode` (post-quantization) |
| `frag_counts` | (B, 4096) | uint8 histogram, sum ≤ 51 | `frag_inds_to_counts(codebook_inds)` |
| `feature_vector` | (B, 512) | float (continuous, **pre-VQ**) | `VQGAE.encode` second return value |
| `atoms_vectors` | (B, 51, 512) | float (post-VQ) | `VQGAE.encode` first return value |

The current decoder pipeline consumes **`codebook_inds`**. Generation works by proposing fragment compositions (most easily as `frag_counts`), reconstructing an ordering with `OrderingNetwork`, and decoding. See `examples/bring_your_own_optimizer.py` for swapping the published GA for any external optimizer.

## Intended use

- De novo molecular generation under constraints (active learning, inverse QSAR, scaffold-constrained design).
- Validity rates on decoded molecules are typically 10–30%; downstream filtering (PAINS / ring-size / valence sanity) is expected.
- This is research-quality code; not a clinical or production tool.

## License

MIT.

## Citation

```bibtex
@article{akhmetshin2023construction,
  title={Construction of order-independent molecular fragments space with vector quantised graph autoencoder},
  author={Akhmetshin, Timur and Lin, Albert and Madzhidov, Timur and Varnek, Alexandre},
  journal={ChemRxiv},
  publisher={Cambridge Open Engage},
  year={2023},
  doi={10.26434/chemrxiv-2023-5zmvw}
}
```
