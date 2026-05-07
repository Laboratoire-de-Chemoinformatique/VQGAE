"""End-to-end VQGAE Tubulin inverse-QSAR pipeline (CPU smoke run).

Loads bundled fragment-count features + RF QSAR classifier, runs a 5-gen
PyGAD GA over the 4096-d fragment-count space, decodes hits, applies
structure filters, prints a final summary block.

Run from anywhere:
    uv run python examples/run_tubulin_e2e.py
"""

from __future__ import annotations

import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pygad

from VQGAE import (
    VQGAE,
    OrderingNetwork,
    decode_population,
    filter_molecule,
    tanimoto_kernel,
)

# Resolve absolute paths relative to this script so it works from anywhere.
SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT.parents[1]  # .../VQGAE/VQGAE
DATA_DIR = PROJECT_ROOT / "data" / "tubulin"


def _resolve_ckpt(env_var: str, hf_filename: str) -> Path:
    """Resolve a checkpoint path. Priority:
    1. $env_var (explicit override)
    2. ../VQGAE_app/saved_model/<filename>  (sibling-of-repo lab layout)
    3. ./data/checkpoints/<filename>        (in-repo cache)
    4. HF hub download from `tagirshin/VQGAE`
    """
    if env_var in os.environ:
        return Path(os.environ[env_var])
    candidates = [
        PROJECT_ROOT.parent / "VQGAE_app" / "saved_model" / hf_filename,
        PROJECT_ROOT / "data" / "checkpoints" / hf_filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    # last resort — pull from HF
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download("tagirshin/VQGAE", hf_filename))


VQGAE_CKPT = _resolve_ckpt("VQGAE_CKPT", "vqgae.ckpt")
ONN_CKPT = _resolve_ckpt("ONN_CKPT", "ordering_network.ckpt")

BATCH = 500
NUM_GENS = 5
RANDOM_SEED = 42


def main() -> int:
    captured_warnings: list[warnings.WarningMessage] = []

    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")

        t0 = time.perf_counter()

        # ------------------------------------------------------------------ #
        # 1. Load bundled data + RF classifier
        # ------------------------------------------------------------------ #
        npz = np.load(DATA_DIR / "tubulin_qsar_class_train_data_vqgae.npz")
        X = npz["x"]
        Y = npz["y"]
        with open(DATA_DIR / "rf_class_train_tubulin.pickle", "rb") as fh:
            rf_model = pickle.load(fh)

        print(f"X: {X.shape}   Y: {Y.shape}   active fraction: {float(Y.mean()):.3f}")
        rf_score_attr = getattr(rf_model, "best_score_", None)
        if rf_score_attr is not None:
            print(f"RF balanced accuracy (10-fold CV): {rf_score_attr:.3f}")

        # ------------------------------------------------------------------ #
        # 2. Load checkpoints from local disk
        # ------------------------------------------------------------------ #
        print(f"Loading VQGAE encode head from {VQGAE_CKPT} ...")
        VQGAE.load_from_checkpoint(
            str(VQGAE_CKPT), task="encode", batch_size=BATCH, map_location="cpu"
        ).eval()
        print(f"Loading VQGAE decode head from {VQGAE_CKPT} ...")
        vqgae_dec = VQGAE.load_from_checkpoint(
            str(VQGAE_CKPT), task="decode", batch_size=BATCH, map_location="cpu"
        ).eval()
        print(f"Loading OrderingNetwork from {ONN_CKPT} ...")
        ordering_model = OrderingNetwork.load_from_checkpoint(
            str(ONN_CKPT), batch_size=BATCH, map_location="cpu"
        ).eval()

        # ------------------------------------------------------------------ #
        # 2b. Decode round-trip sanity check on the 603 training rows
        # ------------------------------------------------------------------ #
        print("Decode round-trip on the 603 training fragment-count rows ...")
        rt_t0 = time.perf_counter()
        rt_err = None
        rt_valid = 0
        try:
            _, rt_validity, _ = decode_population(
                X, vqgae_dec, ordering_model, batch_size=BATCH
            )
            rt_valid = int(sum(rt_validity))
        except Exception as exc:
            rt_err = repr(exc)
        rt_secs = time.perf_counter() - rt_t0
        if rt_err is None:
            print(
                f"  round-trip ok: {rt_valid}/{len(X)} valid "
                f"({rt_valid / len(X) * 100:.1f}%) in {rt_secs:.1f}s"
            )
        else:
            print(f"  round-trip FAILED: {rt_err}")

        # ------------------------------------------------------------------ #
        # 3. Fitness function (mirrors notebook 02)
        # ------------------------------------------------------------------ #
        def fitness_func_batch(_ga, solutions, _solutions_indices):
            counts = np.array(solutions)
            rf = rf_model.predict_proba(counts)[:, 1]
            size_penalty = np.where(counts.sum(-1) < 18, -1.0, 0.0)
            dissim = 1 - tanimoto_kernel(counts, X).max(-1)
            dissim += np.where(dissim == 0, -5, 0)
            return (0.5 * rf + 0.3 * dissim + size_penalty).tolist()

        # ------------------------------------------------------------------ #
        # 4. PyGAD GA
        # ------------------------------------------------------------------ #
        initial_pop = X
        num_parents = int(initial_pop.shape[0] * 0.33 // 10 * 10)  # ~190
        keep_parents = int(num_parents * 0.66 // 10 * 10)  # ~120
        print(
            f"PyGAD: gens={NUM_GENS}  pop={initial_pop.shape[0]}  "
            f"parents_mating={num_parents}  keep_parents={keep_parents}"
        )

        ga = pygad.GA(
            fitness_func=fitness_func_batch,
            initial_population=initial_pop,
            num_genes=initial_pop.shape[-1],
            fitness_batch_size=BATCH,
            num_generations=NUM_GENS,
            num_parents_mating=num_parents,
            parent_selection_type="rws",
            crossover_type="single_point",
            mutation_type="adaptive",
            mutation_percent_genes=[10, 5],
            save_best_solutions=False,
            save_solutions=True,
            keep_elitism=0,
            keep_parents=keep_parents,
            suppress_warnings=True,
            random_seed=RANDOM_SEED,
            gene_type=int,
        )
        ga_t0 = time.perf_counter()
        ga.run()
        ga_secs = time.perf_counter() - ga_t0
        print(f"GA finished in {ga_secs:.1f}s")

        # ------------------------------------------------------------------ #
        # 5. Dedupe + rescore
        # ------------------------------------------------------------------ #
        solutions = list({tuple(s) for s in ga.solutions})
        print(f"{len(solutions)} unique candidates")

        rf_scores: list[float] = []
        sim_scores: list[float] = []
        for i in range(0, len(solutions), 100):
            chunk = np.array(solutions[i : i + 100])
            rf_scores.extend(rf_model.predict_proba(chunk)[:, 1].tolist())
            sim_scores.extend(tanimoto_kernel(chunk, X).max(-1).tolist())
        sc_df = pd.DataFrame({"rf_score": rf_scores, "similarity_score": sim_scores})

        # ------------------------------------------------------------------ #
        # 6. Coarse filter
        # ------------------------------------------------------------------ #
        chosen = sc_df[(sc_df["similarity_score"] < 0.95) & (sc_df["rf_score"] > 0.5)]
        print(f"{len(chosen)} candidates pass coarse filters")

        # ------------------------------------------------------------------ #
        # 7. Decode chosen candidates
        # ------------------------------------------------------------------ #
        decoded_valid = 0
        mols: list = []
        validity: list = []
        ordering_scores: list = []

        if len(chosen) > 0:
            ids = chosen.index.to_list()
            chosen_solutions = np.array([solutions[i] for i in ids])
            mols, validity, ordering_scores = decode_population(
                chosen_solutions, vqgae_dec, ordering_model, batch_size=100
            )
            decoded_valid = int(sum(validity))
            print(f"{decoded_valid}/{len(validity)} decode to valid molecules")

        # ------------------------------------------------------------------ #
        # 8. Structure filters
        # ------------------------------------------------------------------ #
        final_records: list[dict] = []
        if len(chosen) > 0:
            for mol, ok, rf_sc, sim_sc, ord_sc, _idx in zip(
                mols,
                validity,
                chosen.rf_score,
                chosen.similarity_score,
                ordering_scores,
                chosen.index,
            ):
                if not ok:
                    continue
                try:
                    keep = filter_molecule(mol)
                except Exception:
                    keep = False
                if not keep:
                    continue
                final_records.append(
                    {
                        "smiles": str(mol),
                        "rf_score": float(rf_sc),
                        "similarity_score": float(sim_sc),
                        "ordering_score": float(ord_sc),
                    }
                )

        total_secs = time.perf_counter() - t0
        captured_warnings = list(wlist)

    # ---------------------------------------------------------------------- #
    # 9. Final summary block
    # ---------------------------------------------------------------------- #
    print()
    print("=" * 64)
    print("FINAL SUMMARY")
    print("=" * 64)
    print(f"GA generations           : {NUM_GENS}")
    print(f"GA wall time             : {ga_secs:.2f}s")
    print(f"Total wall time          : {total_secs:.2f}s")
    print(f"Decode round-trip ok     : {rt_err is None}  (valid {rt_valid}/{len(X)})")
    print(f"Unique GA solutions      : {len(solutions)}")
    print(f"Coarse-filter survivors  : {len(chosen)}")
    if len(chosen) > 0:
        validity_rate = decoded_valid / max(len(validity), 1) * 100
        print(
            f"Decoded valid            : {decoded_valid}/{len(validity)} "
            f"({validity_rate:.1f}%)"
        )
    else:
        print("Decoded valid            : 0/0 (no chosen candidates)")
    print(f"Structure-filter survivors: {len(final_records)}")
    if final_records:
        rf_mean = float(np.mean([r["rf_score"] for r in final_records]))
        print(f"Mean RF score (survivors): {rf_mean:.3f}")
        print("Example survivor SMILES:")
        for rec in final_records[:5]:
            print(
                f"  {rec['smiles']}   rf={rec['rf_score']:.3f}  "
                f"sim={rec['similarity_score']:.3f}  ord={rec['ordering_score']:.3f}"
            )
    else:
        print("Mean RF score (survivors): n/a")
        print("Example survivor SMILES  : (none)")

    # ---------------------------------------------------------------------- #
    # 10. Non-trivial warnings worth flagging
    # ---------------------------------------------------------------------- #
    interesting: dict[tuple[str, str], int] = {}
    boring_substrings = (
        "deprecated",
        "TypedStorage",
        "torch.load",
        "weights_only",
        "FutureWarning",
        "pkg_resources",
    )
    for w in captured_warnings:
        msg = str(w.message)
        cat = w.category.__name__
        if any(s.lower() in msg.lower() for s in boring_substrings) and cat in {
            "DeprecationWarning",
            "FutureWarning",
            "UserWarning",
        }:
            # still count, but de-prioritise
            key = (cat, msg.splitlines()[0][:120] + " [boring]")
        else:
            key = (cat, msg.splitlines()[0][:160])
        interesting[key] = interesting.get(key, 0) + 1
    print()
    print("WARNINGS (deduped, top 15):")
    for (cat, msg), n in sorted(interesting.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  [{cat} x{n}] {msg}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
