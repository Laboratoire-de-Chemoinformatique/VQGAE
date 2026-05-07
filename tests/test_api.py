"""Fast, model-free tests for the public VQGAE API.

These tests do **not** load model checkpoints and do **not** hit the network.
They verify:
  - Public symbols import.
  - Pure utility functions behave correctly.
  - The bundled tubulin data fixture loads with expected shapes.
  - Structure filters classify hand-picked good/bad molecules.

Run with:
    uv run --extra cpu --extra examples pytest tests/ -v
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "tubulin"


# ---------------------------------------------------------------------------
# 1. Public API surface
# ---------------------------------------------------------------------------


def test_public_api_imports():
    import VQGAE

    expected = {
        "VQGAE",
        "OrderingNetwork",
        "encode_smiles",
        "decode_population",
        "vqgae_encode_dataset",
        "vqgae_encode_mols",
        "tanimoto_kernel",
        "frag_inds_to_counts",
        "frag_counts_to_inds",
        "restore_order",
        "decode_molecules",
        "filter_molecule",
        "create_chem_graph",
    }
    assert expected.issubset(set(VQGAE.__all__)), (
        f"missing exports: {expected - set(VQGAE.__all__)}"
    )
    for name in expected:
        assert getattr(VQGAE, name) is not None, f"{name} resolves to None"


def test_chython_drop_in_for_cgrtools():
    """Confirm chython exposes everything VQGAE source imports."""
    from chython import smiles
    from chython.containers import MoleculeContainer

    m = smiles("CCO")
    m.canonicalize()
    assert isinstance(m, MoleculeContainer)


# ---------------------------------------------------------------------------
# 2. tanimoto_kernel
# ---------------------------------------------------------------------------


def test_tanimoto_kernel_identity():
    from VQGAE import tanimoto_kernel

    x = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.int64)
    K = tanimoto_kernel(x, x)
    np.testing.assert_allclose(np.diag(K), [1.0, 1.0])
    # rows are orthogonal -> off-diagonal Tanimoto is 0
    assert K[0, 1] == 0.0
    assert K[1, 0] == 0.0


def test_tanimoto_kernel_known_value():
    from VQGAE import tanimoto_kernel

    x = np.array([[1, 1, 1, 0]])  # |x|^2 = 3
    y = np.array([[1, 1, 0, 1]])  # |y|^2 = 3, <x,y> = 2
    # Tanimoto = 2 / (3 + 3 - 2) = 2/4 = 0.5
    K = tanimoto_kernel(x, y)
    assert K.shape == (1, 1)
    np.testing.assert_allclose(K[0, 0], 0.5)


def test_tanimoto_kernel_zero_rows():
    from VQGAE import tanimoto_kernel

    x = np.zeros((1, 4), dtype=np.int64)
    y = np.zeros((1, 4), dtype=np.int64)
    # both rows zero -> NaN handled -> 0
    K = tanimoto_kernel(x, y)
    assert K.shape == (1, 1)
    assert K[0, 0] == 0.0


# ---------------------------------------------------------------------------
# 3. frag_inds <-> frag_counts roundtrip (utils versions, torch-aware)
# ---------------------------------------------------------------------------


def test_frag_counts_roundtrip():
    from VQGAE import frag_counts_to_inds, frag_inds_to_counts

    # known fragment indices for two molecules, padded with -1 to length 51
    mol_a = [3, 3, 7, 12, 12, 12] + [-1] * 45
    mol_b = [0, 1, 2, 3] + [-1] * 47
    inds = np.array([mol_a, mol_b], dtype=np.int64)

    counts = frag_inds_to_counts(inds, num_frags=4096)
    assert counts.shape == (2, 4096)
    assert counts[0, 3] == 2
    assert counts[0, 7] == 1
    assert counts[0, 12] == 3
    assert counts[1, 0] == 1
    assert counts.sum(axis=-1).tolist() == [6, 4]

    # roundtrip back to indices (sorted ascending — matches frag_counts_to_inds order)
    inds_back = frag_counts_to_inds(counts, max_atoms=51).numpy()
    # canonical recovered ordering: ascending fragment id, repeated by count
    assert inds_back[0, :6].tolist() == [3, 3, 7, 12, 12, 12]
    assert inds_back[0, 6] == -1
    assert inds_back[1, :4].tolist() == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# 4. Tubulin data fixture
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (DATA_DIR / "tubulin_qsar_class_train_data_vqgae.npz").exists(),
    reason="tubulin fixture not available",
)
def test_tubulin_fixture_shapes():
    npz = np.load(DATA_DIR / "tubulin_qsar_class_train_data_vqgae.npz")
    X, Y = npz["x"], npz["y"]
    assert X.shape == (603, 4096), f"X shape {X.shape}"
    assert Y.shape == (603,), f"Y shape {Y.shape}"
    # all sums within max_atoms cap
    assert X.sum(-1).max() <= 51
    # binary label
    assert set(np.unique(Y).tolist()).issubset({0, 1})


@pytest.mark.skipif(
    not (DATA_DIR / "rf_class_train_tubulin.pickle").exists(),
    reason="tubulin RF not available",
)
def test_tubulin_rf_predicts():
    with open(DATA_DIR / "rf_class_train_tubulin.pickle", "rb") as fh:
        rf = pickle.load(fh)
    npz = np.load(DATA_DIR / "tubulin_qsar_class_train_data_vqgae.npz")
    X = npz["x"]
    proba = rf.predict_proba(X[:5])
    assert proba.shape == (5, 2)
    np.testing.assert_allclose(proba.sum(-1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. Structure filter
# ---------------------------------------------------------------------------


def test_filter_molecule_passes_drug():
    from chython import smiles

    from VQGAE import filter_molecule

    m = smiles("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    m.canonicalize()
    m.kekule()
    m.thiele()
    assert filter_molecule(m) is True


def test_filter_molecule_rejects_macrocycle():
    from chython import smiles

    from VQGAE import filter_molecule

    # 12-membered all-carbon ring — outside the 4..8 ring-size window
    m = smiles("C1CCCCCCCCCCC1")
    m.canonicalize()
    assert filter_molecule(m) is False


def test_filter_molecule_rejects_peroxide():
    from chython import smiles

    from VQGAE import filter_molecule

    m = smiles("CCOOC")  # has peroxide bond
    m.canonicalize()
    assert filter_molecule(m) is False
