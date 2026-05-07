# OrderingNetwork — what it is, why it exists

## The problem it solves

VQGAE's encoder is permutation-invariant: the codebook indices for a molecule
are returned in *sorted* order (largest first), but the decoder is
permutation-sensitive — a single set of `frag_counts` could decode to many
different (or no) valid molecule depending on how the indices are arranged
across the `(B, 51)` slot grid.

The OrderingNetwork is a separate small model that, given a multiset of
codebook indices, predicts the canonical ordering the VQGAE decoder was
trained on. Without it, the decoder produces nonsense most of the time.

## How it's used

```python
from VQGAE import frag_counts_to_inds, restore_order, decode_molecules

inds   = frag_counts_to_inds(counts, max_atoms=51)      # (B, 51) torch int64, -1 padded
canon, ord_scores = restore_order(inds, ordering_model) # solves an LSAP per molecule
mols, valid       = decode_molecules(canon, vqgae_dec)
```

`decode_population` wraps these three steps so you don't need to think
about it:

```python
mols, valid, ord_scores = decode_population(counts, vqgae_dec, ordering_model)
```

The `ord_scores` are the mean per-atom assignment confidence from the LSAP
solution; values close to 1.0 mean the network was very sure about the
ordering, values closer to 0 mean the proposed multiset is far from the
training distribution and the decoded molecule is likely garbage. A common
filter is `ordering_score > 0.7` (used in the published Tubulin run).

## Architecture

`VQGAE.OrderingNetwork`:
- Embedding: per codebook entry, `vector_dim = 512`, `vq_embeddings = 4096`
- 8 multi-head attention layers, 16 heads each
- Outputs a `(B, max_atoms, max_atoms)` matrix of soft assignment scores
- We then run a linear sum assignment problem (Hungarian algorithm) to pick
  the best permutation

Total ~27 M parameters — about a third the size of VQGAE itself.

## When you need to retrain it

You need to retrain the OrderingNetwork **whenever the VQGAE codebook
changes**. The codebook is fixed during VQGAE training but it's specific to
that VQGAE instance, so:

| Did you train a new VQGAE? | Retrain ONN? |
|-----------------------------|----------------|
| Just changed batch_size / lr / dropout — same codebook | No |
| Trained on a different SDF (new codebook) | **Yes** |
| Forked the codebase and changed `vq_embeddings` | **Yes** |
| Using the published `tagirshin/VQGAE` checkpoints | No, use the published ONN |

## Retraining recipe

Same as `train_from_scratch.md` step 4:

```bash
# 1. After training VQGAE, encode the training set to safetensors.
uv run vqgae_encode -c configs/vqgae_encode.yaml \
                    --ckpt_model_file ./your_vqgae.ckpt
# -> ./outputs/encoded_codebook_inds.safetensors

# 2. Train the OrderingNetwork on those indices.
uv run onn_train fit -c configs/ordering_network_training.yaml
```

The ONN learns by predicting the random-shuffle inverse: at training time it
sees a permuted set of codebook indices and is asked to recover the canonical
order. So no extra labels are needed — the safetensors file is the only input.

Typical training: 1000 epochs, batch_size=1000, ~few hours on a single GPU
for a 1.6 M-molecule ChEMBL set.
