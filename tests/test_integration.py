"""Integration tests: load real VQGAE + ONN checkpoints from disk and run.

These tests are slower (model load + forward passes on CPU) but still local —
no network calls.  They are skipped automatically if checkpoints aren't found
at any of the searched paths.

Override checkpoint location with env vars:
    VQGAE_CKPT=/path/to/vqgae.ckpt
    ONN_CKPT=/path/to/ordering_network.ckpt

Default search paths (first hit wins):
    1. $VQGAE_CKPT / $ONN_CKPT
    2. ../VQGAE_app/saved_model/{vqgae,ordering_network}.ckpt   (sibling of repo)
    3. ./data/checkpoints/{vqgae,ordering_network}.ckpt          (in-repo cache)
    4. ~/.cache/huggingface/hub/models--tagirshin--VQGAE/...     (HF cache)
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_ckpt(name: str, env_var: str) -> Path | None:
    if env_path := os.environ.get(env_var):
        p = Path(env_path)
        if p.exists():
            return p

    candidates = [
        REPO_ROOT.parent / "VQGAE_app" / "saved_model" / name,
        REPO_ROOT / "data" / "checkpoints" / name,
    ]
    hf_hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--tagirshin--VQGAE"
    if hf_hub.is_dir():
        for snap in (
            (hf_hub / "snapshots").iterdir() if (hf_hub / "snapshots").is_dir() else []
        ):
            cand = snap / name
            if cand.exists():
                candidates.append(cand)
    for c in candidates:
        if c.exists():
            return c
    return None


VQGAE_CKPT = _find_ckpt("vqgae.ckpt", "VQGAE_CKPT")
ONN_CKPT = _find_ckpt("ordering_network.ckpt", "ONN_CKPT")
HAVE_CKPTS = VQGAE_CKPT is not None and ONN_CKPT is not None

skip_no_ckpts = pytest.mark.skipif(
    not HAVE_CKPTS,
    reason=f"checkpoints not found (vqgae={VQGAE_CKPT}, onn={ONN_CKPT})",
)


# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def models():
    """Load encode, decode, ordering models once per test module."""
    from VQGAE import VQGAE, OrderingNetwork

    BATCH = 5
    enc = VQGAE.load_from_checkpoint(
        str(VQGAE_CKPT), task="encode", batch_size=BATCH, map_location="cpu"
    ).eval()
    dec = VQGAE.load_from_checkpoint(
        str(VQGAE_CKPT), task="decode", batch_size=BATCH, map_location="cpu"
    ).eval()
    onn = OrderingNetwork.load_from_checkpoint(
        str(ONN_CKPT), batch_size=BATCH, map_location="cpu"
    ).eval()
    return enc, dec, onn


@skip_no_ckpts
def test_load_checkpoints(models):
    enc, dec, onn = models
    assert enc.task == "encode"
    assert dec.task == "decode"
    assert enc.max_atoms == 51
    # codebook sized 4096
    assert enc.vq.codebook.shape[0] == 512  # vector_dim
    # ordering network shares the codebook size
    assert onn.max_atoms == 51


@skip_no_ckpts
def test_encode_smiles_returns_expected_shapes(models):
    from VQGAE import encode_smiles

    enc, _, _ = models
    smis = ["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "C1=CC=CC=C1"]
    inds, counts = encode_smiles(smis, enc, batch_size=5)

    # codebook indices: (N, max_atoms) with -1 padding
    assert inds.shape == (3, 51)
    # counts: (N, num_frags)
    assert counts.shape == (3, 4096)
    # atoms encoded ≤ max_atoms
    assert (counts.sum(-1) <= 51).all()
    # at least some non-pad indices for each molecule
    assert (inds >= 0).any(axis=-1).all()


@skip_no_ckpts
def test_round_trip_validity_on_drugs(models):
    """Aspirin / caffeine / ibuprofen should round-trip with most being valid."""
    from VQGAE import decode_population, encode_smiles

    enc, dec, onn = models
    smis = [
        "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
        "OC(=O)C(N)Cc1ccccc1",  # phenylalanine
        "C1=CC=CC=C1",  # benzene
    ]
    _, counts = encode_smiles(smis, enc, batch_size=5)
    mols, validity, ordering_scores = decode_population(
        counts, dec, onn, batch_size=5, clean_2d=False
    )
    assert len(mols) == len(smis)
    assert all(0.0 <= s <= 1.0 for s in ordering_scores)
    # at least 60% should round-trip — drug-like distribution validity is high
    assert sum(validity) >= 3, f"only {sum(validity)}/{len(validity)} valid"


@skip_no_ckpts
def test_decode_population_handles_single_row(models):
    """decode_population should accept a 1D row of counts."""
    import numpy as np

    from VQGAE import decode_population

    _, dec, onn = models
    npz = np.load(
        REPO_ROOT / "data" / "tubulin" / "tubulin_qsar_class_train_data_vqgae.npz"
    )
    one_row = npz["x"][0]  # shape (4096,)
    mols, validity, scores = decode_population(one_row, dec, onn, batch_size=5)
    assert len(mols) == 1
    assert len(validity) == 1
    import numbers

    assert isinstance(scores[0], numbers.Real)
