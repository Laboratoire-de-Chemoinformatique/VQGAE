# Vector Quantized Graph-based AutoEncoder (VQGAE)

This repository is the official implementation of the **Vector Quantized Graph-based AutoEncoder**.

- **Preprint:** [ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/6439662408c86922fff8d6e5)
- **Pretrained weights:** [tagirshin/VQGAE](https://huggingface.co/tagirshin/VQGAE) on the Hugging Face Hub
- **Live demo:** the inverse-QSAR Genetic Algorithm pipeline runs on
  [Hugging Face Spaces](https://huggingface.co/spaces/tagirshin/VQGAE)

## Tutorials (`notebooks/`)

| Notebook | What it shows |
|---|---|
| `01_quickstart.ipynb` | Encode + decode round-trip on a few SMILES, ~30 lines. |
| `02_inverse_qsar_tubulin.ipynb` | Full inverse-QSAR pipeline (encode → RF → GA → decode → filter). Reproduces the published Tubulin demo. |
| `03_bring_your_own_optimizer.ipynb` | Same Tubulin objective with PyGAD GA, Optuna TPE, and random search side-by-side. Template for plugging in any external optimizer. |

## Installation (uv, recommended)

The project ships with [uv](https://docs.astral.sh/uv/) sources for CPU and CUDA builds of PyTorch:

```bash
git clone https://github.com/Laboratoire-de-Chemoinformatique/VQGAE.git
cd VQGAE
uv sync --extra cpu --extra examples       # CPU on Mac/Linux
# or:
uv sync --extra cu126 --extra examples     # CUDA 12.6
uv sync --extra cu128 --extra examples     # CUDA 12.8
```

Open the tutorials with:

```bash
uv run jupyter lab notebooks/
```

### Pip (alternative)

```bash
pip install "vqgae[cpu,examples] @ git+https://github.com/Laboratoire-de-Chemoinformatique/VQGAE.git"
```

Available extras: `cpu`, `cu126`, `cu128` (mutually exclusive — pick one), `ga`,
`bo`, `notebooks`, `app`, `examples` (= `ga + bo + notebooks`).

<details>
<summary><strong>Legacy install (conda/manual fallback)</strong></summary>

Use this only if `uv sync` and the pip-with-extras paths above don't work for
your environment (typically: old CUDA drivers, restricted clusters, no `uv`).

#### Conda lockfile

```bash
git clone https://github.com/Laboratoire-de-Chemoinformatique/VQGAE.git
cd VQGAE
conda install --channel=conda-forge --name=base conda-lock      # if not installed
conda env create --name vqgae_env --file vqgae_gpu.yml
conda activate vqgae_env
pip install .
```

#### Fully manual

If conda drivers don't match your machine, install the stack by hand. The tool
was originally tested with CUDA 11.8 + Pytorch 2.0; it works on newer versions
of both — see `pyproject.toml` for the currently-tested constraints.

```bash
# 1. PyTorch (pick the CUDA version matching your driver, or cpu)
pip3 install torch --extra-index-url https://download.pytorch.org/whl/${CUDATORCH}
#    where ${CUDATORCH} ∈ {cpu, cu118, cu121, cu126, cu128}
#    docs: https://pytorch.org/get-started/locally/

# 2. PyTorch Geometric (must match the torch+cuda combo above)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
    -f https://data.pyg.org/whl/torch-${TORCH}+${CUDAPYG}.html
#    docs: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

# 3. Lightning + AdaBelief
pip install "pytorch-lightning>=2.3" "adabelief-pytorch>=0.2.1"

# 4. VQGAE itself
pip install .
```

</details>

## Python API

Top-level imports cover the common use cases:

```python
from huggingface_hub import hf_hub_download
from VQGAE import (
    VQGAE, OrderingNetwork,
    encode_smiles, decode_population,
    tanimoto_kernel, filter_molecule,
)

# load
vqgae_ckpt = hf_hub_download("tagirshin/VQGAE", "vqgae.ckpt")
onn_ckpt   = hf_hub_download("tagirshin/VQGAE", "ordering_network.ckpt")
enc = VQGAE.load_from_checkpoint(vqgae_ckpt, task="encode", batch_size=8, map_location="cpu").eval()
dec = VQGAE.load_from_checkpoint(vqgae_ckpt, task="decode", batch_size=8, map_location="cpu").eval()
onn = OrderingNetwork.load_from_checkpoint(onn_ckpt, batch_size=8, map_location="cpu").eval()

# encode -> 4096-dim integer fragment-count latent
codebook_inds, counts = encode_smiles(["CC(=O)Oc1ccccc1C(=O)O", "C1=CC=CC=C1"], enc)

# round-trip decode
mols, validity, ordering_scores = decode_population(counts, dec, onn)
```

The 4096-dim integer histogram (`counts`, `sum ≤ 51`) is what every published
optimizer (PyGAD GA, NSGA-II, …) searches over. `tanimoto_kernel` and
`filter_molecule` are the same helpers used in the Tubulin paper.

## Command-line interface

```bash
vqgae_train  fit -c configs/vqgae_training.yaml
vqgae_encode     -c configs/vqgae_encode.yaml
vqgae_decode     -c configs/vqgae_decode.yaml
onn_train    fit -c configs/ordering_network_training.yaml
vqgae_default_config --task train      # emits a default config
```

See [`docs/cli.md`](docs/cli.md) for every config knob, and
[`docs/train_from_scratch.md`](docs/train_from_scratch.md) for the
SDF → VQGAE → encode → ONN chain end-to-end.

## Documentation

The [`docs/`](docs/) folder covers everything tutorials don't:

- [`cli.md`](docs/cli.md) — config knobs for every CLI command
- [`latent_shapes.md`](docs/latent_shapes.md) — what `encode()` returns and which tensor your optimizer should target
- [`ordering_network.md`](docs/ordering_network.md) — why a second model exists and when to retrain it
- [`train_from_scratch.md`](docs/train_from_scratch.md) — full retraining recipe
- [`your_own_qsar.md`](docs/your_own_qsar.md) — plug your own activity data into the inverse-QSAR pipeline
- [`fragment_inspection.md`](docs/fragment_inspection.md) — what each codebook fragment represents chemically

## Contributing

Contributions are welcome, in the form of issues or pull requests.

If you have a question or want to report a bug, please submit an issue.

To contribute with code to the project, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the remote branch: `git push`
5. Create the pull request.

## Citation

Please make sure to cite this work if you find it useful:

```bibtex
@article{akhmetshin2023construction,
  title={Construction of order-independent molecular fragments space with vector quantised graph autoencoder},
  author={Akhmetshin, Timur and Lin, Albert and Madzhidov, Timur and Varnek, Alexandre},
  journal={ChemRxiv},
  publisher={Cambridge Open Engage},
  year={2023},
  note={This content is a preprint and has not been peer-reviewed.},
  doi={10.26434/chemrxiv-2023-5zmvw}
}
```

## Copyright

* [Tagir Akhmetshin ](tagirshin@gmail.com)
* [Arkadii Lin](arkadiyl18@gmail.com)
* [Timur Madzhidov](tmadzhidov@gmail.com)
* [Alexandre Varnek](varnek@unistra.fr)