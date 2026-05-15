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


# ---------------------------------------------------------------------------
# 6. RDKit interop helpers (smiles_to_mol / rdkit_to_mol / mol_to_rdkit)
# ---------------------------------------------------------------------------

# A handful of complex real-world drugs picked to exercise stereo, fused
# polycycles, macrocycles, and sugars. SMILES are RDKit-canonical, which is
# the dialect that most often gives chython trouble.
_COMPLEX_DRUGS = {
    "taxol": "CC1=C2[C@@]([C@]([C@H]([C@@H]3[C@]4([C@H](OC4)C[C@@H]([C@]3(C(=O)[C@@H]2OC(=O)C)C)O)OC(=O)C)OC(=O)c5ccccc5)(C[C@@H]1OC(=O)[C@H](O)[C@@H](NC(=O)c6ccccc6)c7ccccc7)O)(C)C",
    "vincristine": "CCC1(CC2CC(c3c(CCN(C2)C1)c4ccccc4[nH]3)(C(=O)OC)C5=C6C7(C8C(C(=O)OC)(c9c(c%10ccccc%10n9C=O)CC8)C(C7)O)CC(O6)(O)C5)O",
    "erythromycin": "CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]3O[C@H](C)C[C@@H]([C@H]3O)N(C)C)[C@](C)(O)C[C@@H](C)C(=O)[C@H](C)[C@@H](O)[C@]1(C)O",
    "imatinib": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc4nccc(n4)c5cccnc5",
    "atorvastatin": "CC(C)c1c(C(=O)Nc2ccccc2)c(c3ccccc3)c(c4ccc(F)cc4)n1CC[C@H](O)C[C@H](O)CC(=O)O",
}


@pytest.mark.parametrize("name,smi", list(_COMPLEX_DRUGS.items()))
def test_smiles_to_mol_complex_drugs(name, smi):
    """`smiles_to_mol` parses RDKit-canonical SMILES of complex drugs."""
    from VQGAE import smiles_to_mol

    m = smiles_to_mol(smi)
    # 30+ heavy atoms for every drug above; check we got a non-trivial graph
    assert len(m) >= 30, f"{name}: expected >=30 atoms, got {len(m)}"


@pytest.mark.parametrize("name,smi", list(_COMPLEX_DRUGS.items()))
def test_rdkit_to_mol_round_trip(name, smi):
    """`mol_to_rdkit` ∘ `rdkit_to_mol` preserves the canonical chython SMILES."""
    from rdkit import Chem

    from VQGAE import mol_to_rdkit, rdkit_to_mol

    rd = Chem.MolFromSmiles(smi)
    assert rd is not None, f"RDKit failed to parse {name}"
    m1 = rdkit_to_mol(rd)
    rd2 = mol_to_rdkit(m1)
    m2 = rdkit_to_mol(rd2)
    assert str(m1) == str(m2), (
        f"{name}: canonical chython SMILES differs after round-trip\n  m1: {m1}\n  m2: {m2}"
    )


def test_smiles_to_mol_no_fallback_propagates():
    """With fallback_to_rdkit=False, a chython parse error must propagate."""
    from VQGAE import smiles_to_mol

    with pytest.raises(Exception):  # noqa: B017 — chython raises a few different exception types
        smiles_to_mol("not a valid smiles", fallback_to_rdkit=False)


def test_smiles_to_mol_unparseable_raises_value_error():
    """A SMILES neither chython nor RDKit can parse raises ValueError."""
    from VQGAE import smiles_to_mol

    with pytest.raises(ValueError, match="could not parse SMILES"):
        smiles_to_mol("ZZZZZ()(()(((((")


# ---------------------------------------------------------------------------
# 7. Sharded preprocessing pipeline (rdkit + datamol + safetensors)
# ---------------------------------------------------------------------------


SDF_FIXTURE = DATA_DIR / "colchicine_scaffold.sdf"


@pytest.mark.skipif(not SDF_FIXTURE.exists(), reason="colchicine SDF fixture missing")
def test_preprocess_to_shards_sdf_round_trip(tmp_path):
    """End-to-end: SDF -> shards -> ShardedVQGAEDataset yields correct count + shape."""
    from VQGAE.preprocessing import ShardedVQGAEDataset, preprocess_to_shards

    summary = preprocess_to_shards(
        SDF_FIXTURE,
        tmp_path,
        shard_size=50,
        max_atoms=51,
        n_jobs=2,
        props_mode="rdkit",
        chunk_size=50,
    )
    assert summary["n_kept"] == summary["total_molecules"]
    assert summary["n_skipped"] == 0
    assert summary["n_shards"] >= 1
    assert (tmp_path / "index.json").exists()

    ds = ShardedVQGAEDataset(tmp_path, shuffle=False)
    assert len(ds) == summary["total_molecules"]
    n = 0
    first = None
    for data in ds:
        if first is None:
            first = data
        n += 1
    assert n == len(ds)
    # Per-molecule tensors are unpadded on read (so PyG `Batch.from_data_list`
    # produces correctly-sized node-level batches): mol_size rows, not max_atoms.
    assert first.x.shape[0] == int(first.mol_size)
    assert first.x.shape[1] == 11
    assert first.atoms_types.shape == (int(first.mol_size),)
    assert first.edge_index.shape[0] == 2
    assert first.edge_attr.shape[0] == first.edge_index.shape[1]
    # rdkit-computed class_prop has 8 entries clipped to DEFAULT_PROP_BINS
    assert tuple(first.class_prop.shape) == (8,)


@pytest.mark.skipif(not SDF_FIXTURE.exists(), reason="colchicine SDF fixture missing")
def test_preprocess_to_shards_smi_path(tmp_path):
    """SMILES file (.smi) goes through the same pipeline."""
    from chython.files import SDFRead

    from VQGAE.preprocessing import ShardedVQGAEDataset, preprocess_to_shards

    smi_path = tmp_path / "test.smi"
    with SDFRead(str(SDF_FIXTURE)) as inp, open(smi_path, "w") as out:
        for i, mol in enumerate(inp):
            if i >= 8:
                break
            out.write(str(mol) + "\n")

    out_dir = tmp_path / "shards"
    summary = preprocess_to_shards(
        smi_path,
        out_dir,
        shard_size=20,
        n_jobs=1,
        props_mode="rdkit",
        chunk_size=20,
    )
    assert summary["n_kept"] >= 5  # some may be filtered for various reasons
    ds = ShardedVQGAEDataset(out_dir, shuffle=False)
    assert len(ds) == summary["total_molecules"]


def test_iter_smiles_smi_format(tmp_path):
    """iter_smiles handles whitespace-delimited .smi with optional ID column."""
    from VQGAE.preprocessing import iter_smiles

    p = tmp_path / "x.smi"
    p.write_text(
        "CCO ethanol\n"
        "c1ccccc1 benzene\n"
        "# this is a comment, skip me\n"
        "\n"  # blank line, also skipped
        "CC(=O)O acetic_acid\n"
    )
    rows = list(iter_smiles(p))
    assert rows == [
        ("CCO", "ethanol"),
        ("c1ccccc1", "benzene"),
        ("CC(=O)O", "acetic_acid"),
    ]


def test_compute_rdkit_props_shape_and_clip():
    """compute_rdkit_props returns 8 ints; clip_props_to_bins enforces ranges."""
    from rdkit import Chem

    from VQGAE.preprocessing import (
        DEFAULT_PROP_BINS,
        clip_props_to_bins,
        compute_rdkit_props,
    )

    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    props = compute_rdkit_props(mol)
    assert len(props) == 8
    clipped = clip_props_to_bins(props)
    for value, max_bin in zip(clipped, DEFAULT_PROP_BINS):
        assert 0 <= value < max_bin
