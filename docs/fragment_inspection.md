# What does each codebook fragment represent?

VQGAE learns a 4096-entry codebook of atom-environment vectors. Each entry
("fragment") is a 512-d continuous vector that the encoder snaps to during
quantization. To make sense of a generated histogram you sometimes need to
ask: *what does fragment 1070 actually look like in real molecules?*

The runnable version of this is `notebooks/06_fragment_inspection.ipynb`.

This is the non-GTM portion of the lab's `fragment_analysis.ipynb` — the
GTM visualization needs the proprietary ISIDA-GTM binary, which we don't
require here. Same insights, plain Python.

## Pipeline

```
SDF -> VQGAE.encode -> per-molecule {atom_idx: fragment_id}
     -> invert: fragment_occurence[fragment_id][mol_idx] = [atom_idxs]
     -> for each fragment, characterize the chemical environment of those atoms
```

## Step 1 — Build `mols_frags`

```python
from chython.files import SDFRead
from VQGAE import VQGAE
from VQGAE.preprocessing import MolDataset
from torch_geometric.loader.dataloader import DataLoader as PYGDataLoader
from huggingface_hub import hf_hub_download
from collections import defaultdict
import numpy as np

vqgae_enc = VQGAE.load_from_checkpoint(
    hf_hub_download("tagirshin/VQGAE", "vqgae.ckpt"),
    task="encode", batch_size=50, map_location="cpu",
).eval()

dataset = MolDataset(max_atoms=51, molecules_file="./my_mols.sdf")
loader  = PYGDataLoader(dataset, batch_size=50)

mols_frags: list[dict[int, int]] = []
for batch in loader:
    sizes = batch.ptr[1:] - batch.ptr[:-1]
    _, _, vq_indices = vqgae_enc.encode(batch)
    vq_indices = vq_indices.cpu().numpy()
    for i, n in enumerate(sizes.tolist()):
        mols_frags.append(
            {atom_i + 1: frag_i for atom_i, frag_i in enumerate(vq_indices[i][:n].tolist())}
        )
```

Each entry is `{atom_index_1based: codebook_id}` for one molecule. The
`+1` matches chython's 1-based atom numbering.

## Step 2 — Invert into `fragment_occurence`

```python
fragment_occurence = defaultdict(lambda: defaultdict(list))
for mol_i, mf in enumerate(mols_frags):
    for atom_i, frag_i in mf.items():
        fragment_occurence[frag_i][mol_i].append(atom_i)
fragment_occurence = {f: dict(v) for f, v in fragment_occurence.items()}
print(f"distinct codebook entries used: {len(fragment_occurence)} / 4096")
```

Typically only ~1200 codebook entries are used by any given dataset. The
rest of the 4096 slots are unused for that data distribution — totally fine,
the codebook was trained on a different (broader) set.

## Step 3 — Atom-environment extractor

Given a chosen atom in a molecule, walk the bond graph outward and tabulate
each shell as a sorted tuple of `(atomic_symbol, hybridization, in_ring)`:

```python
def atom_env_extraction(mol, central_atom: int, max_env: int = 7):
    seen, env = set(), []
    frontier = [central_atom]
    for _ in range(max_env):
        shell = []
        next_frontier = []
        for n in frontier:
            if n in seen:
                continue
            atom = mol.atom(n)
            shell.append((atom.atomic_symbol, atom.hybridization, atom.in_ring))
            next_frontier.extend(mol._bonds[n])
        env.append(tuple(sorted(shell)))
        seen.update(frontier)
        frontier = next_frontier
    return tuple(env)
```

Two atoms with the same `env[:3]` (sphere depth 3) are considered the same
"chemical context".

## Step 4 — Pick a fragment, see what it represents

```python
frag_id = 1070   # change me
mols_ids = fragment_occurence[frag_id]
print(f"fragment {frag_id} appears in {len(mols_ids)} molecules in this dataset")

env_border = 3
unique = defaultdict(list)
with SDFRead("./my_mols.sdf", indexable=True) as inp:
    for mol_id, atom_ids in mols_ids.items():
        molecule = inp[mol_id]
        env = atom_env_extraction(molecule, atom_ids[0])[:env_border]
        unique[env].append(mol_id)

unique = dict(sorted(unique.items(), key=lambda kv: -len(kv[1])))
print(f"  {len(unique)} distinct atom-environment patterns at depth {env_border}")
for n, (pattern, mol_ids) in enumerate(unique.items()):
    if n >= 5: break
    print(f"  {len(mol_ids):>5d}  {pattern}")
```

Sample output for fragment 1070 over a 42 k-molecule slice:

```
fragment 1070 appears in 42394 molecules in this dataset
  5884 distinct atom-environment patterns at depth 3
   1839  ((('C', 1, True),), (('C', 1, True), ('C', 1, True), ('O', 1, False)), ...)
   1424  ((('C', 1, True),), (('C', 1, False), ('C', 1, True), ('O', 1, True)),  ...)
    659  ((('N', 1, True),), (('C', 1, False), ('C', 2, True),  ('C', 2, True)), ...)
   ...
```

So fragment 1070 isn't one chemical motif — it's a continuous neighbourhood
in the codebook embedding space, snapped to by atoms with broadly similar
environments. The most common one (1839 hits) is "ring carbon with two ring
carbons + one non-ring oxygen at depth 1" — i.e., a phenol/anisole-style
substructure.

## What this is good for

- **Sanity-checking the codebook**: a fragment that's used by < 10 molecules
  in your data is a long-tail entry; treat its appearances in generated
  candidates with caution.
- **Designing fragment masks**: when you want to fix a scaffold (tutorial 04),
  this is how you discover which codebook ids actually show up in the
  scaffold's atoms.
- **Diagnosing decode failures**: if `decode_population` is producing too many
  invalid molecules, look at which fragment ids dominate the failures and
  inspect their environments — they may all correspond to a hypervalent or
  charge-ambiguous atom that the decoder isn't trained on.

## What this is not

- A SMARTS pattern: each fragment maps to many real environments; don't
  treat it as a deterministic substructure.
- A drop-in replacement for ECFP: ECFP is hash-based and exact;
  fragment_occurence here is learned and approximate.
