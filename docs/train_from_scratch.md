# Train VQGAE + OrderingNetwork from scratch

Use this when you want a model adapted to a different molecular distribution
than ChEMBL drug-likes (e.g., natural products, fragments, peptidomimetics).

## Prerequisites

- An SDF of training molecules (we used 1.6 M ChEMBL records). At minimum
  a few hundred thousand for a useful codebook.
- A validation SDF (~10 % of train).
- A GPU. The published model trained for 100 epochs at `batch_size=500`,
  finishing in ~4 days on a single A6000.
- The training data needs **`meta` annotations** for the multi-target
  classification head used during VQGAE training. The configs expect these
  property keys (numeric `class` labels):

  ```
  Hetero Atom Count
  acceptorcount
  donorcount
  Chiral center count
  Ring count
  Hetero ring count
  Rotatable bond count
  Aromatic ring count
  ```

  If your SDF doesn't have them, either compute them with RDKit /
  ChemAxon and write them in, or trim the `class_categories` and
  `properties_names` fields in the training config (and matching
  `class_categories` in `model.py`).

- All molecules must have ≤ 51 heavy atoms (`max_atoms`). Drop bigger ones
  upstream.

## Step 1 — Train VQGAE

```bash
# 1. Edit the config: point at your SDFs and tmp/log folders.
vim configs/vqgae_training.yaml
# - data.path_train_predict
# - data.path_val
# - data.tmp_folder
# - trainer.devices, trainer.max_epochs, model.batch_size, etc.

# 2. Train.
uv run vqgae_train fit -c configs/vqgae_training.yaml
```

Outputs:
- `vqgae_model_checkpoint.dirpath` → `vqgae.ckpt` (the encoder + decoder + codebook)
- `trainer.logger` → CSV training metrics

The codebook (4096 entries × 512-d) is initialized at random and updated by
EMA inside the `VectorQuantizer`. It's frozen *after* training.

## Step 2 — Encode the training set into a safetensors file

The OrderingNetwork doesn't see molecular graphs, only the codebook indices
that the trained VQGAE produces for each molecule. So you need to run a
prediction pass over the whole training set.

```bash
# Edit configs/vqgae_encode.yaml, in particular:
#   data.path_train_predict       (your training SDF)
#   ckpt_model_file               (the .ckpt from Step 1)
#   encoder_writer.output_dir     (where safetensors will be written)
uv run vqgae_encode -c configs/vqgae_encode.yaml
```

Output: `<output_dir>/<output_name>_codebook_inds.safetensors` containing
a single tensor `codebook` of shape `(N, max_atoms)` with `-1` padding.

## Step 3 — Train the OrderingNetwork

```bash
# Edit configs/ordering_network_training.yaml:
#   data.input_file               (the safetensors from Step 2)
#   ordering_network_model_checkpoint.dirpath
uv run onn_train fit -c configs/ordering_network_training.yaml
```

Outputs `ordering_network.ckpt`. This is what `decode_population` will use
at generation time.

## Step 4 (optional) — Sanity check

```python
import torch
from VQGAE import VQGAE, OrderingNetwork, encode_smiles, decode_population

vqgae_enc = VQGAE.load_from_checkpoint("./vqgae.ckpt", task="encode", batch_size=10).eval()
vqgae_dec = VQGAE.load_from_checkpoint("./vqgae.ckpt", task="decode", batch_size=10).eval()
onn       = OrderingNetwork.load_from_checkpoint("./ordering_network.ckpt", batch_size=10).eval()

smis = ["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
_, counts = encode_smiles(smis, vqgae_enc, batch_size=10)
mols, valid, scores = decode_population(counts, vqgae_dec, onn, batch_size=10)
for s, m, ok in zip(smis, mols, valid):
    print(s, "->", m if ok else "(invalid)")
```

If validity drops below ~50 % on drug-likes, something's off — either the
ONN didn't see enough epochs, or the safetensors in step 2 was generated
from an out-of-sync VQGAE checkpoint.

## Step 5 (optional) — Decode from saved indices

If you want a CLI-only path to verify the decode side:

```bash
# configs/vqgae_decode.yaml expects a safetensors with key "codebook".
uv run vqgae_decode -c configs/vqgae_decode.yaml
```

Writes a reconstructed SDF.

## Compute & wall time references

Numbers from the published 1.6 M-molecule ChEMBL training on a single A6000:

| Step | Wall time |
|------|-----------|
| 1. VQGAE training (100 epochs, batch_size=500) | ~4 days |
| 2. Full-set encoding (predict pass) | ~30 min |
| 3. ONN training (1000 epochs, batch_size=1000) | ~10 h |

For a smaller exploratory dataset (say, 50 k molecules, 20 epochs) all
three steps fit in a single afternoon.
