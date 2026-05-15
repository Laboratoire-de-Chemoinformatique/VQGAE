"""Sharded safetensors preprocessing for very large training corpora.

Pipeline:
    SMILES / SDF (streaming line iterator)
        -> rdkit Mol
        -> dm.parallelized(preprocess_rdkit)        # per-mol, top-level fn
        -> tensors batched into N=shard_size groups
        -> safetensors.save_file(...)

Read pipeline:
    ShardedVQGAEDataset (IterableDataset, DDP-aware)
        -> safe_open(shard, framework="pt")
        -> get_slice(...) for x / atoms_types / edge_index / edge_attr
        -> yield torch_geometric.data.Data per molecule

Why iterable + safetensors slices: a 5000-shard / 500M-mol corpus is
~550 GB, well beyond RAM. Slice-reads keep memory bounded to roughly one
batch worth of tensors. Workers and DDP ranks each get a disjoint subset
of shards, so disk I/O stays sequential per worker.

Optional: :class:`ShardedVQGAEData` integrates with
``torchdata.stateful_dataloader.StatefulDataLoader`` for mid-epoch resume
when ``stateful=True``. Without that flag a stock PyG ``DataLoader`` is
used. ``torchdata`` is an optional dependency.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PYGDataLoader
from tqdm.auto import tqdm

from ..utils import DEFAULT_MAX_ATOMS
from .rdkit import preprocess_rdkit
from .smi import (
    clip_props_to_bins,
    compute_rdkit_props,
    iter_smiles,
)

def preprocess_one(args: tuple) -> dict | None:
    """Joblib worker: parse SMILES (or accept a pre-parsed Mol), preprocess.

    ``args`` is ``(smi_or_mol, props_mode, max_atoms)``. ``props_mode`` is
    one of ``"rdkit"``, ``"sdf-meta"``, ``"none"`` — sdf-meta is a no-op
    here, the caller attaches class_props before calling.
    """
    from rdkit import Chem  # local import: child process

    smi_or_mol, props_mode, max_atoms = args
    mol = Chem.MolFromSmiles(smi_or_mol) if isinstance(smi_or_mol, str) else smi_or_mol
    if mol is None:
        return None

    if props_mode == "rdkit":
        try:
            props = clip_props_to_bins(compute_rdkit_props(mol))
        except Exception:
            return None
    else:
        props = None  # "none" or "sdf-meta" (caller attaches)
    return preprocess_rdkit(mol, max_atoms=max_atoms, class_props=props)


def write_shard(items: list[dict], out_path: str | Path) -> int:
    """Pack a list of preprocess_rdkit dicts into one safetensors file.

    Schema (per shard):
      x            (N, max_atoms, 11)      int8
      atoms_types  (N, max_atoms)          int8        — -1 padded
      mol_size     (N,)                    int8        — heavy-atom count
      class_prop   (N, 8)                  int8        — only if any item has it
      edge_index   (2, total_E)            int32       — concatenated
      edge_attr    (total_E,)              int8        — concatenated
      edge_ptr     (N+1,)                  int32       — slice boundaries
    """
    items = [it for it in items if it is not None]
    if not items:
        return 0
    n = len(items)

    x = torch.from_numpy(np.stack([it["x"] for it in items]))
    atoms_types = torch.from_numpy(np.stack([it["atoms_types"] for it in items]))
    mol_size = torch.tensor([int(it["mol_size"]) for it in items], dtype=torch.int8)

    edge_lengths = [it["edge_index"].shape[1] for it in items]
    edge_ptr = torch.tensor(np.cumsum([0, *edge_lengths]), dtype=torch.int32)
    edge_index = torch.from_numpy(
        np.concatenate([it["edge_index"] for it in items], axis=1)
    )  # int32
    edge_attr = torch.from_numpy(
        np.concatenate([it["edge_attr"] for it in items])
    )  # int8

    out: dict[str, torch.Tensor] = {
        "x": x,
        "atoms_types": atoms_types,
        "mol_size": mol_size,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_ptr": edge_ptr,
    }
    if "class_prop" in items[0]:
        out["class_prop"] = torch.from_numpy(
            np.stack([it["class_prop"] for it in items])
        ).to(torch.int8)

    save_file(out, str(out_path))
    return n


def count_input_records(input_path: str | Path) -> int | None:
    """Cheap pre-count of input records, used as `tqdm`'s `total`.

    - For ``.smi[.gz]`` / ``.csv[.gz]``: count newlines (subtract 1 if the
      first line looks like a CSV header). On a 30 GB Enamine dump this is
      ~1 minute of streamed read; cheap compared to the actual preprocess.
    - For ``.sdf[.gz]``: counting `$$$$` markers requires a full scan, so we
      return ``None`` and let tqdm show count + rate instead of an ETA.
    """
    p = Path(input_path)
    name = p.name.lower()
    if ".sdf" in name:
        return None
    opener = gzip.open if name.endswith(".gz") else open
    with opener(p, "rt") as fh:
        first = fh.readline()
        if not first:
            return 0
        n = 1
        for _ in fh:
            n += 1
    if name.endswith((".csv", ".csv.gz")):
        # Heuristic: if the first line has no whitespace-separable SMILES,
        # treat it as a header.
        first_stripped = first.strip()
        if first_stripped and not any(c in first_stripped[:5] for c in "()[]"):
            n -= 1
    elif first.startswith("#"):
        n -= 1
    return n


def iter_sdf(path: str | Path):
    """Stream rdkit Mols from .sdf / .sdf.gz."""
    from rdkit import Chem

    p = Path(path)
    if p.name.lower().endswith(".sdf.gz"):
        with gzip.open(p, "rb") as fh:
            supp = Chem.ForwardSDMolSupplier(fh, sanitize=True, removeHs=False)
            yield from supp
    else:
        supp = Chem.SDMolSupplier(str(p), sanitize=True, removeHs=False)
        yield from supp


def iter_input(
    input_path: str | Path,
    *,
    smiles_column: str | int = 0,
    id_column: str | int | None = None,
    props_mode: str = "rdkit",
    sdf_meta_keys: tuple[str, ...] | None = None,
):
    """Yield ``(parsed_input, attached_class_props_or_None)`` per molecule.

    ``parsed_input`` is a SMILES string for SMI/CSV inputs and an rdkit
    ``Mol`` for SDF inputs. The Mol path reads class properties before the
    worker fork — joblib can't pickle rdkit's PropDict across processes
    once the supplier closes.
    """
    name = Path(input_path).name.lower()
    if ".sdf" in name:
        for mol in iter_sdf(input_path):
            if mol is None:
                continue
            props = None
            if props_mode == "sdf-meta" and sdf_meta_keys:
                try:
                    props = clip_props_to_bins(
                        [int(float(mol.GetProp(k))) for k in sdf_meta_keys]
                    )
                except (KeyError, ValueError):
                    continue
            yield mol, props
    else:
        for smi, _id in iter_smiles(
            input_path,
            smiles_column=smiles_column,
            id_column=id_column,
        ):
            yield smi, None  # SMILES path always uses props_mode=rdkit/none


def preprocess_to_shards(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    shard_size: int = 100_000,
    max_atoms: int = DEFAULT_MAX_ATOMS,
    n_jobs: int = -1,
    props_mode: str = "rdkit",
    sdf_meta_keys: tuple[str, ...] | None = None,
    smiles_column: str | int = 0,
    id_column: str | int | None = None,
    start_at_shard: int = 0,
    chunk_size: int = 5_000,
) -> dict[str, Any]:
    """End-to-end: stream input, batch, parallel preprocess, write shards.

    Writes ``shard_00000.safetensors``, ``shard_00001.safetensors``, ...
    plus an ``index.json`` summarising shard counts.

    :param chunk_size: how many molecules to send to ``dm.parallelized`` at
        a time. Smaller = lower peak RAM (only ``chunk_size`` Mol objects
        plus their preprocess results in flight); larger = better worker
        utilisation. The default 5000 is a reasonable trade-off.
    """
    import datamol as dm  # lazy: only writer needs datamol

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_idx = start_at_shard
    accum: list[dict] = []
    n_seen = n_kept = n_skipped = 0

    # Pre-count for a meaningful tqdm ETA (None for SDFs — falls back to count+rate).
    total = count_input_records(input_path)
    if total is not None:
        print(f"[input] {total} records in {input_path}")
    pbar = tqdm(
        total=total,
        desc="preprocess",
        unit="mol",
        unit_scale=True,
        smoothing=0.05,
        dynamic_ncols=True,
    )

    def flush() -> None:
        nonlocal shard_idx
        path = out_dir / f"shard_{shard_idx:05d}.safetensors"
        n = write_shard(accum, path)
        if n:
            pbar.write(f"[shard {shard_idx:05d}] wrote {n} mols -> {path.name}")
        accum.clear()
        shard_idx += 1

    def process_chunk(chunk_items: list, chunk_props_: list) -> None:
        """Parallel preprocess a chunk and append survivors to `accum`."""
        nonlocal n_kept, n_skipped
        if not chunk_items:
            return
        results = dm.parallelized(
            preprocess_one,
            [(p, props_mode, max_atoms) for p in chunk_items],
            n_jobs=n_jobs,
            progress=False,
        )
        for r, p in zip(results, chunk_props_):
            if r is None:
                n_skipped += 1
                continue
            if p is not None and "class_prop" not in r:
                r["class_prop"] = np.asarray(p, dtype=np.int8)
            accum.append(r)
            n_kept += 1
            if len(accum) >= shard_size:
                flush()
        # Update the progress bar with stats so users can see the
        # kept/skipped split without re-reading the log.
        pbar.set_postfix(kept=n_kept, skipped=n_skipped, refresh=False)
        pbar.update(len(chunk_items))

    chunk: list = []
    chunk_props: list = []

    iterator = iter_input(
        input_path,
        smiles_column=smiles_column,
        id_column=id_column,
        props_mode=props_mode,
        sdf_meta_keys=sdf_meta_keys,
    )

    for parsed, sdf_props in iterator:
        n_seen += 1
        chunk.append(parsed)
        chunk_props.append(sdf_props)
        if len(chunk) >= chunk_size:
            process_chunk(chunk, chunk_props)
            chunk.clear()
            chunk_props.clear()

    # tail
    process_chunk(chunk, chunk_props)
    if accum:
        flush()
    pbar.close()

    # index.json — total counts + per-shard counts (for fast len, DDP split)
    shards = sorted(out_dir.glob("shard_*.safetensors"))
    counts = []
    for s in shards:
        with safe_open(str(s), framework="pt") as f:
            counts.append(int(f.get_tensor("mol_size").shape[0]))
    index = {
        "schema_version": 1,
        "max_atoms": max_atoms,
        "props_mode": props_mode,
        "n_shards": len(shards),
        "total_molecules": sum(counts),
        "shard_counts": counts,
        "shard_files": [s.name for s in shards],
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))

    summary = {
        "n_seen": n_seen,
        "n_kept": n_kept,
        "n_skipped": n_skipped,
        "n_shards": len(shards),
        "total_molecules": sum(counts),
        "output_dir": str(out_dir),
    }
    print(f"[done] {summary}")
    return summary


class ShardedVQGAEDataset(IterableDataset):
    """Stream PyG ``Data`` objects from a directory of safetensors shards.

    Each worker (and each DDP rank, if distributed) gets a disjoint subset
    of shards via ``rank * num_workers + worker_id``-stride partitioning.
    Within a shard, samples are read via mmap-backed
    ``safe_open.get_slice`` so memory stays bounded to roughly one batch.

    :param shard_dir: directory containing ``shard_*.safetensors`` files
        and an ``index.json`` written by :func:`preprocess_to_shards`.
    :param shuffle: if True, shuffle shard order *and* within-shard order
        every iteration (per-worker independent RNG).
    :param seed: base seed; combined with epoch and worker id to make
        deterministic-but-distinct shuffles per worker.
    """

    def __init__(
        self,
        shard_dir: str | Path,
        *,
        shuffle: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.shards = sorted(self.shard_dir.glob("shard_*.safetensors"))
        if not self.shards:
            raise FileNotFoundError(f"no shards found under {self.shard_dir}")
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        # index.json carries total counts; we expose them via total_molecules
        # for callers that want a count, but deliberately do NOT implement
        # __len__ on this IterableDataset. Lightning's val-existence check
        # (`if dl:` -> `bool(dl)` -> `__len__() > 0`) silently skips the val
        # loop when the reported length doesn't match the number of items
        # actually yielded after worker-stride + drop_last (issues #19624,
        # #10290). Without __len__ the trainer iterates until StopIteration
        # and val runs every epoch as expected.
        idx_file = self.shard_dir / "index.json"
        self.index = json.loads(idx_file.read_text()) if idx_file.exists() else None
        self.total_molecules = (
            int(self.index["total_molecules"]) if self.index else None
        )

    def set_epoch(self, epoch: int) -> None:
        """Lightning calls this between epochs; new epoch -> new shuffle."""
        self.epoch = epoch

    def my_shards(self) -> list[Path]:
        info = get_worker_info()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world = torch.distributed.get_world_size()
        else:
            rank, world = 0, 1
        n_workers = info.num_workers if info else 1
        worker_id = info.id if info else 0
        stride = world * n_workers
        offset = rank * n_workers + worker_id
        return list(self.shards[offset::stride])

    def __iter__(self) -> Iterator[Data]:
        info = get_worker_info()
        worker_id = info.id if info else 0
        rng = np.random.default_rng((self.seed, self.epoch, worker_id))
        shards = self.my_shards()
        if self.shuffle:
            rng.shuffle(shards)

        for path in shards:
            with safe_open(str(path), framework="pt") as f:
                ptr = f.get_tensor("edge_ptr")
                n = ptr.shape[0] - 1
                # mmap-backed slices: only the rows we touch get paged in
                x_view = f.get_slice("x")
                at_view = f.get_slice("atoms_types")
                ms_view = f.get_slice("mol_size")
                ei_view = f.get_slice("edge_index")
                ea_view = f.get_slice("edge_attr")
                # `safe_open` doesn't implement __contains__ / __iter__,
                # so the Pythonic `"class_prop" in f` raises TypeError.
                # Use the explicit keys() call instead.
                shard_keys = set(f.keys())
                cp_view = (
                    f.get_slice("class_prop") if "class_prop" in shard_keys else None
                )

                order = np.arange(n)
                if self.shuffle:
                    rng.shuffle(order)
                for i in order:
                    e0, e1 = int(ptr[i]), int(ptr[i + 1])
                    ms = int(ms_view[i])
                    # x and atoms_types are stored padded to max_atoms; the
                    # encoder + PyG `Batch.from_data_list` expect node-level
                    # tensors of shape (mol_size, ...) — unpad on load so the
                    # batch concat produces the right total node count.
                    data = Data(
                        x=x_view[i][:ms].to(torch.long),
                        edge_index=ei_view[:, e0:e1].to(torch.long),
                        edge_attr=ea_view[e0:e1].to(torch.long),
                        atoms_types=at_view[i][:ms],
                        mol_size=ms,
                    )
                    if cp_view is not None:
                        data.class_prop = cp_view[i]
                    yield data


class ShardedVQGAEData(LightningDataModule):
    """LightningDataModule wrapping :class:`ShardedVQGAEDataset` for train/val.

    Compatible with the model's ``max_atoms`` / ``batch_size`` argument
    linking via ``parser.link_arguments`` in ``cli.py`` — exposes both
    keyword arguments at the top level.

    :param stateful: if True, return ``StatefulDataLoader`` from torchdata.
        Required for mid-epoch resume; needs ``torchdata>=0.10`` installed.
    """

    def __init__(
        self,
        train_dir: str,
        val_dir: str | None = None,
        batch_size: int = 500,
        max_atoms: int = DEFAULT_MAX_ATOMS,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = True,
        shuffle: bool = True,
        stateful: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.max_atoms = max_atoms
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.stateful = stateful
        self.seed = seed
        self._pending_state: dict | None = None

    def make_loader(self, dataset, *, num_workers: int, drop_last: bool):
        if self.stateful:
            try:
                from torchdata.stateful_dataloader import StatefulDataLoader
            except ImportError as e:
                raise ImportError(
                    "stateful=True requires `torchdata>=0.10`. "
                    "Install with `uv add torchdata` or pass stateful=False."
                ) from e
            loader_cls = StatefulDataLoader
        else:
            loader_cls = PYGDataLoader
        # IterableDataset handles its own shuffling; do not pass shuffle=
        return loader_cls(
            dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            persistent_workers=num_workers > 0,
        )

    def train_dataloader(self):
        ds = ShardedVQGAEDataset(self.train_dir, shuffle=self.shuffle, seed=self.seed)
        loader = self.make_loader(
            ds, num_workers=self.num_workers, drop_last=self.drop_last
        )
        if self.stateful and self._pending_state is not None:
            loader.load_state_dict(self._pending_state)
            self._pending_state = None
        return loader

    def val_dataloader(self):
        if self.val_dir is None:
            return None
        ds = ShardedVQGAEDataset(self.val_dir, shuffle=False, seed=self.seed)
        # num_workers=0: workers >> n_shards leaves most workers empty
        # (shards[offset::stride] partitions by stride), and Lightning's
        # val loop short-circuits on the resulting StopIteration.
        # drop_last=True: the VQGAE model pads predictions to a fixed
        # `batch_size` via to_dense_batch, but `to_dense_adj` (target)
        # infers batch from the data — a trailing partial batch crashes
        # compute_bonds_loss with a target/input shape mismatch. Now
        # that __len__ is no longer defined on the dataset, drop_last
        # no longer causes the IterableDataset val-skip bug.
        return PYGDataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def state_dict(self):
        if not self.stateful:
            return {}
        loader = (
            getattr(self.trainer, "train_dataloader", None)
            if hasattr(self, "trainer")
            else None
        )
        if loader is None or not hasattr(loader, "state_dict"):
            return {}
        return {"loader_state": loader.state_dict()}

    def load_state_dict(self, state_dict):
        self._pending_state = state_dict.get("loader_state") if state_dict else None


__all__ = [
    "ShardedVQGAEData",
    "ShardedVQGAEDataset",
    "preprocess_to_shards",
    "write_shard",
]
