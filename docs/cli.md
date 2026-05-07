# CLI commands

`pip install -e .` (or `uv sync`) registers five console-scripts that wrap
the same components used by the notebooks:

| Command | Function | Datamodule | Notes |
|---------|----------|-------------|-------|
| `vqgae_train fit` | Train a fresh VQGAE | `VQGAEData` | Reads SDFs, kekulizes, encodes graph features. |
| `vqgae_encode` | Predict-only pass: SDF → safetensors of codebook indices | `VQGAEData` | Needs a trained `vqgae.ckpt`. |
| `vqgae_decode` | Decode a safetensors of codebook indices back to SDF | `VQGAEVectors` | Needs the same `vqgae.ckpt`. |
| `onn_train fit` | Train an OrderingNetwork on the safetensors from `vqgae_encode` | `VQGAEVectors` | See `ordering_network.md` for theory. |
| `vqgae_default_config --task {train,encode,decode}` | Emit a starter YAML to CWD | – | Helpful when you want a config from scratch. |

All four PyTorch-Lightning commands use `LightningCLI` with
`parser_mode=omegaconf` — pass `-c <config.yaml>` and override anything
on the command line with `--<group>.<key>=<value>`.

Sample configs live in `configs/`. Edit them in place, or copy elsewhere.

## `vqgae_train fit`

Wired through `VQGAE.cli.training_interface` → `TrainingCLI`. Adds two
extra Lightning class args:

- `vqgae_lr_monitor` (`pytorch_lightning.callbacks.LearningRateMonitor`).
  Default: `logging_interval: epoch`. Requires a `trainer.logger`.
- `vqgae_model_checkpoint` (`pytorch_lightning.callbacks.ModelCheckpoint`).
  Default: `mode: min`, `monitor: train_loss`.

The most-touched knobs in `configs/vqgae_training.yaml`:

| Group.key | Default | What it controls |
|-----------|---------|------------------|
| `seed_everything` | 42 | Set this if you care about exact reproducibility. |
| `trainer.accelerator` | `gpu` | Set to `cpu` for sanity-only runs. |
| `trainer.devices` | `[0]` | Which GPU(s). `[0,1]` for DDP across two cards. |
| `trainer.max_epochs` | 100 | The published model used 100. Drop to 5 for smoke. |
| `trainer.precision` | `16` (`16-mixed` on PL ≥ 2.4) | AMP. Set to `32-true` if you see NaN losses. |
| `trainer.gradient_clip_val` | 1.0 | Required — without it the codebook EMA blows up. |
| `model.max_atoms` | 51 | **Don't change** unless you also reset the codebook size. |
| `model.batch_size` | 500 | Memory-bound; tune per GPU. |
| `model.vector_dim` | 512 | Codebook embedding dim. Tied to `embedding_dim` in the ONN config. |
| `model.vq_embeddings` | 4096 | Codebook size. Larger = finer-grained, slower convergence. |
| `model.lr` | 2e-4 | Used with AdaBelief (`betas=(0.9, 0.999)`, `weight_decay=0.01`). |
| `model.dropout` | 0.2 | – |
| `model.shuffle_graph` | `false` | If `true`, atom order is permuted at train time (data augmentation). |
| `model.reparam` | `false` | Enable the VAE-style reparameterization head + KLD term. Off in the published model. |
| `model.class_categories` | `[38, 29, 21, 25, 16, 13, 50, 13]` | Number of bins for each property classification head. Order matters and must match `data.properties_names`. |
| `data.path_train_predict` | – | Training SDF. |
| `data.path_val` | – | Validation SDF. |
| `data.tmp_folder` | – | Where preprocessed `.pt` caches go. Re-used across runs — delete to force re-preprocessing. |
| `data.tmp_name` | – | Prefix for the cache files. |
| `data.properties_names` | – | `dict[meta_key, "class"|"reg"]`. The keys must exist as `mol.meta[...]` in your SDF. |
| `data.num_workers` | 0 | Increase for dataloader parallelism (8–16 typical). |
| `data.pin_memory` | True | – |
| `data.drop_last` | True | – |
| `vqgae_model_checkpoint.dirpath` | – | Where checkpoints land. |
| `vqgae_model_checkpoint.monitor` | `train_loss` | Switch to `val_loss` if you trust your val split. |

Override on the CLI without editing the YAML:

```bash
uv run vqgae_train fit -c configs/vqgae_training.yaml \
    --trainer.max_epochs=5 \
    --model.batch_size=64 \
    --data.path_train_predict=/path/to/my_train.sdf \
    --data.path_val=/path/to/my_val.sdf
```

## `vqgae_encode`

Wired through `encoding_interface` → `EncodingCLI`. Adds:

- `--ckpt_model_file` *(required)*: path to a trained `vqgae.ckpt`.
- `encoder_writer` (`VQGAE.callbacks.EncoderPredictionsWriter`):
  - `output_name` — prefix for the safetensors file
  - `output_dir` — directory
  - `use_chunks` (bool), `chunk_size` (int)

The model config is shared with `vqgae_train` but **must have `task: encode`**.
Most knobs you tuned for training carry over identically; the practical
ones to know are:

| Group.key | Effect |
|-----------|--------|
| `model.batch_size` | Encoding throughput. Larger is better until OOM. |
| `data.path_train_predict` | The SDF to encode (yes, the key name is awkward — it's reused). |
| `data.tmp_folder`, `data.tmp_name` | Preprocessed-cache location. |
| `encoder_writer.use_chunks` | If `true`, writes one safetensors per chunk (good for huge SDFs). |
| `encoder_writer.chunk_size` | Records per chunk. |

```bash
uv run vqgae_encode -c configs/vqgae_encode.yaml \
    --ckpt_model_file=./vqgae.ckpt \
    --data.path_train_predict=./my_mols.sdf \
    --encoder_writer.output_dir=./outputs
```

## `vqgae_decode`

Wired through `decoding_interface` → `DecodingCLI`. Adds:

- `--ckpt_model_file` *(required)*
- `decoder_writer` (`VQGAE.callbacks.DecoderPredictionsWriter`):
  - `output_file` — SDF to write reconstructed molecules to

Model config must have `task: decode`. The data side here is the
`VQGAEVectors` datamodule, which reads a safetensors file with a
`codebook` tensor of shape `(N, max_atoms)`:

| Group.key | Effect |
|-----------|--------|
| `data.input_file` | safetensors with key `"codebook"` (output of `vqgae_encode`). |
| `decoder_writer.output_file` | Reconstructed SDF. |

```bash
uv run vqgae_decode -c configs/vqgae_decode.yaml \
    --ckpt_model_file=./vqgae.ckpt \
    --data.input_file=./outputs/encoded_codebook_inds.safetensors \
    --decoder_writer.output_file=./reconstructed.sdf
```

## `onn_train fit`

Wired through `ordering_interface` → `TrainingONNCLI`. Adds:

- `ordering_network_lr_monitor` (`LearningRateMonitor`)
- `ordering_network_model_checkpoint` (`ModelCheckpoint`,
  defaults `mode: max`, `monitor: val_rec_rate`)

Model is `OrderingNetwork`. Datamodule is `VQGAEVectors` reading the
safetensors from `vqgae_encode`. The published config trained 1000 epochs
at `batch_size=1000`.

| Group.key | Effect |
|-----------|--------|
| `model.max_atoms` | Must match the VQGAE that produced the safetensors (51). |
| `model.vq_embeddings` | Must match the VQGAE codebook size (4096). |
| `model.embedding_dim` | The ONN's own internal embedding dim. The published model uses 512 (= VQGAE `vector_dim`); other values are fine. |
| `model.num_heads`, `model.num_mha_layers` | Architecture knobs for the ONN transformer. |
| `model.dropout`, `model.lr`, `model.init_values` | Standard. |
| `data.input_file` | Safetensors from `vqgae_encode`. |

```bash
uv run onn_train fit -c configs/ordering_network_training.yaml \
    --data.input_file=./outputs/encoded_codebook_inds.safetensors \
    --trainer.max_epochs=1000
```

## `vqgae_default_config`

A click CLI (not Lightning) that emits a starter YAML to your current
directory:

```bash
uv run vqgae_default_config --task train     # writes ./default_vqgae_config_train.yaml
uv run vqgae_default_config --task encode    # writes ./default_vqgae_config_encode.yaml
uv run vqgae_default_config --task decode    # writes ./default_vqgae_config_decode.yaml
```

This is purely a starter — **prefer the curated `configs/*.yaml`** in this
repo for real runs. The default templates are minimal and have a few
deprecated keys (`gpus: 0` instead of `accelerator/devices`).

## Putting it all together

The chain `SDF → vqgae_train → vqgae_encode → onn_train` and how each
command's outputs feed the next is documented narratively in
[`train_from_scratch.md`](train_from_scratch.md).
