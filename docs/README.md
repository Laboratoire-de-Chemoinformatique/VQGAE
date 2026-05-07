# VQGAE Documentation

Short docs covering the things tutorials don't (full retraining flow, latent
representation reference, OrderingNetwork explainer, fragment inspection
recipe).

## Contents

| Doc | What it covers |
|-----|----------------|
| [`cli.md`](cli.md) | Every console-script (`vqgae_train`, `vqgae_encode`, `vqgae_decode`, `onn_train`, `vqgae_default_config`) and the config knobs each one exposes. |
| [`latent_shapes.md`](latent_shapes.md) | Reference table: every tensor `VQGAE.encode` returns, shapes, dtypes, and which one downstream optimizers should target. |
| [`ordering_network.md`](ordering_network.md) | What the auxiliary `OrderingNetwork` does, why it exists, and when you need to retrain it. |
| [`train_from_scratch.md`](train_from_scratch.md) | End-to-end retraining recipe: prepare an SDF → train VQGAE → encode the training set → train OrderingNetwork. |
| [`your_own_qsar.md`](your_own_qsar.md) | Plug your own activity dataset into the inverse-QSAR pipeline (encode → train sklearn → GA / BO → decode). |
| [`fragment_inspection.md`](fragment_inspection.md) | What each codebook fragment actually represents chemically — non-GTM excerpt of the lab's `fragment_analysis.ipynb`. |

## Tutorials (`../notebooks/`)

| Notebook | What it shows |
|----------|----------------|
| `01_quickstart.ipynb` | Encode + decode round-trip on a few SMILES. |
| `02_inverse_qsar_tubulin.ipynb` | Full Tubulin GA pipeline (encode → RF → GA → decode → filter). |
| `03_bring_your_own_optimizer.ipynb` | PyGAD GA vs Optuna TPE vs random search on the same objective. |
| `04_scaffold_constrained_generation.ipynb` | NSGA-II with the colchicine scaffold pinned. |
| `05_train_your_own_qsar.ipynb` | Encode your own SDF, train an sklearn RF on the fragment counts, run a small GA. |
| `06_fragment_inspection.ipynb` | Look up which atom environments a chosen codebook entry represents. |

## Where to start

- New to VQGAE: tutorial 01, then `latent_shapes.md`.
- "I have my own activity data, give me hits": tutorial 05 (it's tutorial 02 with your data instead of Tubulin).
- "I want Bayesian optimization, not GA": tutorial 03.
- "I want scaffold constraints": tutorial 04.
- "I trained my own VQGAE; how do I retrain the OrderingNetwork?": `ordering_network.md`.
- "What does fragment 1070 actually mean?": tutorial 06 + `fragment_inspection.md`.
