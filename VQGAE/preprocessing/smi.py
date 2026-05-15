"""SMILES-first input handling for the sharded preprocessor.

The published ChEMBL training pipeline expects an SDF with eight ChemAxon-
computed property columns in ``mol.meta``. SMILES files (``.smi`` /
``.csv``) carry no such metadata. Two functions in here close that gap:

- :func:`iter_smiles` — streaming line-iterator over ``.smi[.gz]`` /
  ``.csv[.gz]`` so a 30 GB Enamine REAL dump never materialises in memory.
- :func:`compute_rdkit_props` — produce the eight integer property values
  the model's classifier head expects, using rdkit's descriptors.

The rdkit-computed values are *not* bit-for-bit identical to ChemAxon's:
hydrogen-bond definitions and chiral-centre detection differ. Train a fresh
model on rdkit-computed props and the classifier head adapts; mixing the
two during fine-tuning is the only thing to avoid.
"""

from __future__ import annotations

import csv
import gzip
from collections.abc import Iterator
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Lipinski, rdMolDescriptors


def _open_text(path: str | Path):
    """Open ``.gz``-aware in text mode."""
    p = Path(path)
    return gzip.open(p, "rt") if p.suffix == ".gz" else open(p)


def iter_smiles(
    path: str | Path,
    *,
    smiles_column: str | int = 0,
    id_column: str | int | None = None,
    has_header: bool | None = None,
) -> Iterator[tuple[str, str | None]]:
    """Stream ``(smiles, id)`` pairs from a SMILES or CSV file.

    Supported formats (auto-detected by suffix):
      - ``.smi`` / ``.smi.gz``  — whitespace-delimited: ``SMILES [ID]``
      - ``.csv`` / ``.csv.gz``  — comma-delimited; pass ``smiles_column``
        as the column header (string) or 0-based index (int).

    :param has_header: for CSV, force-true to read the first row as
        column names; force-false to ignore the first row and use integer
        indices. ``None`` (default) sniffs.
    """
    p = Path(path)
    name = p.name.lower()
    is_csv = ".csv" in name

    with _open_text(p) as fh:
        if is_csv:
            sample = fh.read(8192)
            fh.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
                detected_header = (
                    has_header
                    if has_header is not None
                    else csv.Sniffer().has_header(sample)
                )
            except csv.Error:
                dialect = csv.excel
                detected_header = bool(has_header)

            reader = (
                csv.DictReader(fh, dialect=dialect)
                if detected_header and isinstance(smiles_column, str)
                else csv.reader(fh, dialect=dialect)
            )

            for row in reader:
                if isinstance(row, dict):
                    smi = row.get(smiles_column)
                    mol_id = row.get(id_column) if id_column else None
                else:
                    if not row:
                        continue
                    idx = int(smiles_column) if isinstance(smiles_column, int) else 0
                    if idx >= len(row):
                        continue
                    smi = row[idx]
                    mol_id = (
                        row[int(id_column)]
                        if id_column is not None and int(id_column) < len(row)
                        else None
                    )
                if smi:
                    yield smi.strip(), (mol_id.strip() if mol_id else None)
        else:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(maxsplit=1)
                yield parts[0], (parts[1].strip() if len(parts) > 1 else None)


def compute_rdkit_props(rdkit_mol: Chem.Mol) -> list[int]:
    """Eight integer descriptors aligned with the published model's class head.

    Returns ``[hetero_atoms, hba, hbd, chiral_centers, ring_count,
    hetero_ring_count, rotatable_bonds, aromatic_ring_count]``. Bins
    are clipped to the model's ``class_categories`` ranges by the writer,
    not here — this function returns raw descriptor values.

    Order and definition match what the published ChEMBL training expected:

    ============================ ====== =====================
    Descriptor                   Bins   rdkit equivalent
    ============================ ====== =====================
    Hetero Atom Count            38     ``a.GetAtomicNum() not in (1, 6)``
    H-bond acceptor count        29     ``Lipinski.NumHAcceptors``
    H-bond donor count           21     ``Lipinski.NumHDonors``
    Chiral centre count          25     ``Chem.FindMolChiralCenters``
    Ring count                   16     ``RingInfo.NumRings``
    Hetero ring count            13     rings with >= 1 non-C atom
    Rotatable bond count         50     ``rdMolDescriptors.CalcNumRotatableBonds``
    Aromatic ring count          13     rings with all atoms aromatic
    ============================ ====== =====================
    """
    rings = rdkit_mol.GetRingInfo().AtomRings()
    hetero = sum(a.GetAtomicNum() not in (1, 6) for a in rdkit_mol.GetAtoms())
    hetero_rings = sum(
        any(rdkit_mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in r) for r in rings
    )
    aromatic_rings = sum(
        all(rdkit_mol.GetAtomWithIdx(i).GetIsAromatic() for i in r) for r in rings
    )
    return [
        hetero,
        Lipinski.NumHAcceptors(rdkit_mol),
        Lipinski.NumHDonors(rdkit_mol),
        len(Chem.FindMolChiralCenters(rdkit_mol, includeUnassigned=True)),
        len(rings),
        hetero_rings,
        rdMolDescriptors.CalcNumRotatableBonds(rdkit_mol),
        aromatic_rings,
    ]


# Default bins matching the published model's `class_categories`. Indexed
# by descriptor position (see compute_rdkit_props docstring).
DEFAULT_PROP_BINS: tuple[int, ...] = (38, 29, 21, 25, 16, 13, 50, 13)


def clip_props_to_bins(
    props: list[int],
    bins: tuple[int, ...] = DEFAULT_PROP_BINS,
) -> list[int]:
    """Clip each property value to ``[0, bins[i] - 2]`` for the classifier head.

    Why ``-2`` and not ``-1``: ``models.VQGAE._get_loss`` does
    ``cross_entropy(pred, true + 1)``. The classifier outputs ``bins[i]``
    logits per target (see ``layers.MultiTargetClassifier``); valid label
    indices are therefore ``[0, bins[i] - 1]``. Index 0 is reserved for
    "missing/-1"; real values ``0..bins[i] - 2`` map to logits ``1..bins[i] - 1``.

    A value at ``bins[i] - 1`` here would shift to ``bins[i]`` and fire
    ``Assertion cur_target < n_classes`` on the GPU at training time.
    """
    return [min(max(p, 0), b - 2) for p, b in zip(props, bins)]


__all__ = [
    "DEFAULT_PROP_BINS",
    "clip_props_to_bins",
    "compute_rdkit_props",
    "iter_smiles",
]
