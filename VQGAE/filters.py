"""RDKit-backed structure filters used in inverse-QSAR pipelines.

Wraps the standard medchem-alert workflows from the lab notebooks
(research/experiments/inverse_qsar/ga_with_fixed_scaffold.ipynb) so users
don't reimplement them.

`rdkit` is a hard dep of this module — import it lazily so users who only
need encode/decode don't pay the import cost.
"""

from __future__ import annotations

from collections.abc import Iterable

from chython.containers import MoleculeContainer

# Medchem alert SMARTS used in the published scaffold-constrained run:
#   - macrocycles (8 .. 17-atom rings)
#   - peroxide and disulfide
#   - tertiary carbocations
#   - heteroatom-heteroatom singles (N-N, N-S)
#   - alkynes
SMARTS_ALERTS: tuple[str, ...] = (
    "[*;r8]",
    "[*;r9]",
    "[*;r10]",
    "[*;r11]",
    "[*;r12]",
    "[*;r13]",
    "[*;r14]",
    "[*;r15]",
    "[*;r16]",
    "[*;r17]",
    "[#8][#8]",  # peroxide
    "[#6;+]",  # carbocation
    "[#16][#16]",  # disulfide
    "[#7;!n][S;!$(S(=O)=O)]",  # N-S (excl. sulfonamides)
    "[#7;!n][#7;!n]",  # N-N (non-aromatic)
    "C#C",  # alkyne
)


def mol_to_rdkit(
    mol: MoleculeContainer, *, compute_2d: bool = False, clear_atom_maps: bool = True
):
    """Convert a chython `MoleculeContainer` to an RDKit `Mol`.

    Wraps the boilerplate that appears in every notebook:
      - copy + kekule (RDKit doesn't accept chython's aromatic order 4)
      - `mol.to_rdkit(...)`
      - optional 2D coordinate computation
      - optional atom-map number reset

    :param compute_2d:        run `rdkit.Chem.AllChem.Compute2DCoords` on the result.
    :param clear_atom_maps:   set every atom's `AtomMapNum` to 0 (the default
                              is True because chython preserves the per-atom
                              numbering and that bleeds into SMILES output).
    """
    from rdkit.Chem import AllChem  # lazy

    tmp = mol.copy()
    tmp.kekule()
    rdkit_mol = tmp.to_rdkit(keep_mapping=False, keep_hydrogens=False)
    if compute_2d:
        AllChem.Compute2DCoords(rdkit_mol, clearConfs=True)
    if clear_atom_maps:
        for atom in rdkit_mol.GetAtoms():
            atom.SetAtomMapNum(0)
    return rdkit_mol


def check_smarts_alerts(rdkit_mol, alerts: Iterable[str] = SMARTS_ALERTS) -> bool:
    """Return True iff the molecule has *no* match against any alert SMARTS.

    A False return means the molecule should be filtered out.
    """
    from rdkit import Chem  # lazy

    for smarts in alerts:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue
        if rdkit_mol.HasSubstructMatch(pattern):
            return False
    return True


def pains_brenk_nih_filter(
    rdkit_mol,
    catalogs: tuple[str, ...] = ("PAINS", "NIH", "BRENK"),
) -> bool:
    """Return True iff the molecule passes (i.e. matches *no* catalog entry).

    Uses RDKit's bundled `FilterCatalog`. Catalog names are
    case-sensitive attributes on `FilterCatalogParams.FilterCatalogs`
    (`PAINS`, `BRENK`, `NIH`, `ZINC`, `CHEMBL`, …).
    """
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams  # lazy

    params = FilterCatalogParams()
    for name in catalogs:
        params.AddCatalog(getattr(FilterCatalogParams.FilterCatalogs, name))
    catalog = FilterCatalog(params)
    return not catalog.HasMatch(rdkit_mol)


def compute_qed_sas(rdkit_mol) -> tuple[float, float]:
    """Return `(QED, SAScore)` for a molecule.

    SAScore lives in RDKit's contrib SA_Score directory and isn't on the
    public path — this helper handles the `sys.path.append` ritual.
    """
    import sys

    from rdkit.Chem import QED, RDConfig  # lazy

    sa_dir = f"{RDConfig.RDContribDir}/SA_Score"
    if sa_dir not in sys.path:
        sys.path.append(sa_dir)
    import sascorer

    return float(QED.qed(rdkit_mol)), float(sascorer.calculateScore(rdkit_mol))


__all__ = [
    "SMARTS_ALERTS",
    "check_smarts_alerts",
    "compute_qed_sas",
    "mol_to_rdkit",
    "pains_brenk_nih_filter",
]
