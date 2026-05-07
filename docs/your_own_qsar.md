# Train your own sklearn QSAR and run inverse design

The Tubulin tutorials (02–04) use a pretrained Random Forest from
`data/tubulin/`. Here's how to substitute your own activity dataset and
re-run the same pipeline. **You do *not* need to retrain VQGAE for this** —
the published checkpoints already give you a useful 4096-d fragment-count
representation.

The end-to-end version of this is `notebooks/05_train_your_own_qsar.ipynb`.
This page is the narrative.

## What you need

- An **SDF** with one molecule per record and a **single label per molecule**.
  - For classification: `meta["activity"] = "0"` or `"1"` (or any int label).
  - For regression: `meta["pIC50"] = "7.32"` etc.
- A few hundred to a few thousand molecules. RFs work well below 5k; below
  300 you're going to fight overfitting whatever you do.

## Pipeline

```
SDF -> VQGAE.encode -> frag_counts (N, 4096) -> sklearn fit -> pickle -> plug into tutorial 02 or 03
```

## Step 1 — Encode your SDF

```python
import numpy as np
from huggingface_hub import hf_hub_download
from VQGAE import VQGAE, vqgae_encode_dataset, frag_inds_to_counts

vqgae_enc = VQGAE.load_from_checkpoint(
    hf_hub_download("tagirshin/VQGAE", "vqgae.ckpt"),
    task="encode", batch_size=50, map_location="cpu",
).eval()

codebook_inds = vqgae_encode_dataset("./my_actives.sdf", vqgae_enc, batch_size=50)
X = frag_inds_to_counts(codebook_inds).astype(np.int64)   # (N, 4096)
print("X:", X.shape)
```

`vqgae_encode_dataset` walks the SDF in batches and returns the codebook
indices. Internally it uses `VQGAE.preprocessing.MolDataset` which:

- Skips molecules with `> 51` heavy atoms (counted, not raised).
- Skips chython `ValenceError`s (rare hypervalent N/S).
- Kekulizes everything before featurizing.

## Step 2 — Read your labels

```python
from chython.files import SDFRead

Y = []
with SDFRead("./my_actives.sdf", indexable=True) as inp:
    for mol in inp:
        Y.append(int(mol.meta["activity"]))   # adjust key + cast for your case
Y = np.array(Y)
assert len(Y) == X.shape[0], "label/feature row count mismatch"
```

## Step 3 — Train + persist sklearn

The Tubulin RF was a `GridSearchCV` over `n_estimators ∈ [100..500]` and
`max_features ∈ {sqrt, log2}`, scored by 10-fold CV balanced accuracy:

```python
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV

cv = KFold(n_splits=10, shuffle=True, random_state=42)
grid = {
    "n_estimators": list(range(100, 550, 50)),
    "max_features": ["sqrt", "log2"],
}
rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    grid,
    scoring="balanced_accuracy",  # or "r2" for regression
    n_jobs=-1,
    cv=cv,
)
rf.fit(X, Y)

print(f"best params: {rf.best_params_}")
print(f"CV score:    {rf.best_score_:.3f}")

with open("./my_qsar_rf.pickle", "wb") as fh:
    pickle.dump(rf, fh)
```

For regression, swap `RandomForestClassifier` → `RandomForestRegressor`
and `"balanced_accuracy"` → `"r2"`.

> **Sklearn version note**: pickles are not portable across major sklearn
> versions. Train and run the inverse pipeline in the same environment. The
> bundled `data/tubulin/_retrain_rf.py` is the fallback if you ever
> upgrade sklearn and the old pickle stops loading.

## Step 4 — Plug into the GA pipeline

The fitness function in tutorial 02 looks like this:

```python
def fitness_func_batch(_ga, solutions, _idx):
    counts = np.array(solutions)
    rf_score = rf_model.predict_proba(counts)[:, 1]      # <- your model
    size_pen = np.where(counts.sum(-1) < 18, -1.0, 0.0)
    dissim = 1 - tanimoto_kernel(counts, X).max(-1)      # diversity vs your training set
    dissim += np.where(dissim == 0, -5, 0)
    return (0.5 * rf_score + 0.3 * dissim + size_pen).tolist()
```

Replace `rf_model` and `X` with your own. Everything else (PyGAD config,
decode + filter steps) carries over verbatim.

For regression, `rf_score = rf_model.predict(counts)` and adjust the
weights / sign so fitness goes up with desired direction.

## Step 5 — Use a different optimizer

If you want Bayesian optimization instead of GA, point `notebooks/03_bring_your_own_optimizer.ipynb`'s
Optuna `objective` at your scoring function. The latent space is the same
4096-d integer histogram either way.

```python
import optuna
def objective(trial):
    counts = sample_candidate(trial)              # see tutorial 03
    return float(my_score(counts).item())
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
)
study.optimize(objective, n_trials=400)
```

## Common pitfalls

- **Tiny dataset (< 300 mols)**: RF tends to overfit. Consider Tanimoto-distance
  k-NN instead, or add a Bayesian RF / Gaussian Process. Or stay with RF but
  reserve a hard-held-out test set and trust the CV score lightly.
- **Severe class imbalance**: use `class_weight="balanced"`, score by
  `balanced_accuracy` or `roc_auc`, and consider over-sampling the minority
  class.
- **Mixed activity types** (IC50, Ki, %inh): convert to a single uniform
  signal before training (e.g., classify all >= 6.5 pIC50 as active).
