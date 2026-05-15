import json
from abc import ABC
from pathlib import Path

import torch
from chython.files import SDFWrite
from pytorch_lightning.callbacks import BasePredictionWriter, Callback, ModelCheckpoint
from safetensors.torch import save_file
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as PYGDataLoader
from torch_geometric.utils import to_dense_adj, to_dense_batch

from VQGAE.utils import decoded_mol

from .utils import create_chem_graph


class EncoderPredictionsWriter(BasePredictionWriter, ABC):
    def __init__(
        self,
        output_name: str,
        output_dir: str,
        use_chunks: bool = True,
        chunk_size: int = 400,
    ):
        super().__init__()
        self.use_chunks = use_chunks
        self.output_dir = Path(output_dir).resolve(strict=True)
        self.output_name = output_name
        if use_chunks:
            self.chunk_size = chunk_size
            self.chunk_id = 0
            self.batch_counter = 1
            self.features_tmp = []
            self.codebook_tmp = []

    def write_output(self):
        feature_dim = self.features_tmp[0].shape[-1]
        codebook_dim = self.codebook_tmp[0].shape[-1]
        output = {
            "features": torch.reshape(
                torch.stack(self.features_tmp), (-1, feature_dim)
            ),
            "codebook": torch.reshape(
                torch.stack(self.codebook_tmp), (-1, codebook_dim)
            ),
        }
        if self.use_chunks:
            output_file = self.output_dir.joinpath(
                f"{self.output_name}_chunk_{self.chunk_id:03}.safetensors"
            )
        else:
            output_file = self.output_dir.joinpath(f"{self.output_name}.safetensors")
        save_file(output, str(output_file))
        if self.use_chunks:
            del self.features_tmp
            del self.codebook_tmp
            self.features_tmp = []
            self.codebook_tmp = []

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        feature_vector = prediction[1].cpu()
        codebook_inds = prediction[2].cpu()
        self.features_tmp.append(feature_vector)
        self.codebook_tmp.append(codebook_inds)
        if self.use_chunks:
            if self.chunk_size <= self.batch_counter:
                self.write_output()
                self.chunk_id += 1
                self.batch_counter = 0
            else:
                self.batch_counter += 1

    def on_predict_epoch_end(self, trainer, pl_module):
        if self.features_tmp:
            self.write_output()


class DecoderPredictionsWriter(BasePredictionWriter, ABC):
    def __init__(self, output_file: str):
        super().__init__()
        self.output_file = SDFWrite(output_file)

    def write_file(self, prediction):
        for j in range(prediction[0].shape[0]):
            molecule = create_chem_graph(
                prediction[0][j],
                prediction[1][j],
                int(prediction[2][j]),
            )
            self.output_file.write(molecule)

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        decoder_pred = (
            prediction[0].cpu().numpy(),
            prediction[1].cpu().numpy(),
            prediction[2].cpu().numpy(),
        )
        self.write_file(decoder_pred)

    def on_predict_epoch_end(self, trainer, pl_module):
        self.output_file.close()


# Stage-1 / Stage-2 reconstruction checks run on_train_end against the best
# checkpoint to isolate per-stage reconstruction quality from full e2e drift.
#
# Terminology (project-wide):
#   stage 1 = encoder -> VQ -> decoder, atoms+bonds in encoder-native slot order
#             (no ONN, no categorical head). What ``ReconstructionRate`` measures
#             during training.
#   stage 2 = OrderingNetwork (ONN): reorder VQ codebook indices so a fixed
#             VQGAE decoder produces a canonical/valid graph.
#   full    = encoder -> VQ -> ONN -> decoder, end-to-end inference.

def stage1_forward(model, batch):
    """Stage 1: encoder -> VQ -> decoder with ground-truth atom ordering."""
    t_atoms, atoms_mask = to_dense_batch(
        batch.atoms_types.long(),
        batch.batch,
        max_num_nodes=model.max_atoms,
        batch_size=model.batch_size,
    )
    t_bonds = to_dense_adj(
        batch.edge_index.long(),
        batch.batch,
        batch.edge_attr.long(),
        max_num_nodes=model.max_atoms,
    )
    mol_sizes = atoms_mask.long().sum(-1)
    atoms_vectors, _, _ = model.encoder(batch, mol_sizes)
    b, n, d = atoms_vectors.shape
    av, _, _ = model.vq(atoms_vectors.reshape(-1, d))
    p_atoms_logits, p_bonds_logits = model.decoder(av.reshape(b, n, d))
    return p_atoms_logits.argmax(-1), p_bonds_logits.argmax(-3), t_atoms, t_bonds, mol_sizes


def per_mol_metrics(p_atoms, p_bonds, t_atoms, t_bonds, mol_sizes):
    """Reduce a batch into per-mol counts."""
    totals = {
        "n_mols": 0,
        "atoms_correct": 0,
        "atoms_total": 0,
        "bonds_correct": 0,
        "bonds_total": 0,
        "decoded_valid": 0,
        "graph_match": 0,
        "all_atoms_correct": 0,
        "all_bonds_correct": 0,
    }
    p_atoms_np = p_atoms.cpu().numpy()
    p_bonds_np = p_bonds.cpu().numpy()
    t_atoms_np = t_atoms.cpu().numpy()
    t_bonds_np = t_bonds.cpu().numpy()
    sizes_np = mol_sizes.cpu().numpy()
    for i, size in enumerate(sizes_np):
        size = int(size)
        if size < 2:
            continue
        totals["n_mols"] += 1
        pa, ta = p_atoms_np[i, :size], t_atoms_np[i, :size]
        atoms_eq = (pa == ta).sum()
        totals["atoms_correct"] += int(atoms_eq)
        totals["atoms_total"] += size
        if atoms_eq == size:
            totals["all_atoms_correct"] += 1

        iu = torch.triu_indices(size, size, offset=1)
        pb = p_bonds_np[i][iu[0].numpy(), iu[1].numpy()]
        tb = t_bonds_np[i][iu[0].numpy(), iu[1].numpy()]
        bonds_eq = (pb == tb).sum()
        n_pairs = size * (size - 1) // 2
        totals["bonds_correct"] += int(bonds_eq)
        totals["bonds_total"] += n_pairs
        if bonds_eq == n_pairs:
            totals["all_bonds_correct"] += 1

        pred_mol = decoded_mol(p_atoms_np[i], p_bonds_np[i], size)
        gt_mol = decoded_mol(t_atoms_np[i], t_bonds_np[i], size)
        if pred_mol is not None:
            totals["decoded_valid"] += 1
        if pred_mol is not None and gt_mol is not None and pred_mol == gt_mol:
            totals["graph_match"] += 1
    return totals


def to_rates(totals: dict) -> dict:
    n = max(totals["n_mols"], 1)
    return {
        "n_mols": totals["n_mols"],
        "atom_rec_rate": totals["atoms_correct"] / max(totals["atoms_total"], 1),
        "bond_rec_rate": totals["bonds_correct"] / max(totals["bonds_total"], 1),
        "decoded_valid": totals["decoded_valid"] / n,
        "graph_match": totals["graph_match"] / n,
        "per_mol_all_atoms_correct": totals["all_atoms_correct"] / n,
        "per_mol_all_bonds_correct": totals["all_bonds_correct"] / n,
    }


def val_dataset(trainer):
    """Return the trainer's val dataset (or None)."""
    src = trainer.val_dataloaders
    if src is None and trainer.datamodule is not None:
        src = trainer.datamodule.val_dataloader()
    if isinstance(src, (list, tuple)):
        src = src[0]
    return src.dataset if src is not None else None


def fresh_val_loader(trainer, batch_size: int, *, pyg: bool = True):
    """Single-process loader over the val dataset, for gate-checks running
    inside ``on_train_end``. Bypasses the multi-worker val loader to avoid
    shard-stride empties on small val sets.
    """
    dataset = val_dataset(trainer)
    if dataset is None:
        return None
    loader_cls = PYGDataLoader if pyg else TorchDataLoader
    return loader_cls(dataset, batch_size=batch_size, num_workers=0, drop_last=True)


def resolve_log_dir(trainer) -> Path:
    """Writable directory for the JSON dump, falling back to cwd."""
    for attr in ("log_dir", "default_root_dir"):
        d = getattr(trainer, attr, None)
        if d:
            return Path(d)
    return Path.cwd()


def log_gate_metrics(trainer, prefix: str, rates: dict) -> None:
    """Log numeric gate-check metrics via the trainer's logger. ``pl_module.log``
    is disallowed inside ``on_train_end``, so we go through the logger directly.
    """
    if trainer.logger is None:
        return
    payload = {
        f"{prefix}/{k}": float(v)
        for k, v in rates.items()
        if isinstance(v, (int, float))
    }
    if payload:
        trainer.logger.log_metrics(payload, step=trainer.global_step)


def load_best_ckpt_into(trainer, pl_module) -> str | None:
    """Load the trainer's best (or last) ModelCheckpoint weights into
    ``pl_module`` in-place; returns the loaded path, or None when no
    checkpoint was recorded.
    """
    best_path: str | None = None
    for cb in trainer.callbacks or []:
        if isinstance(cb, ModelCheckpoint):
            best_path = cb.best_model_path or cb.last_model_path or None
            if best_path:
                break
    if not best_path or not Path(best_path).is_file():
        return None
    ckpt = torch.load(best_path, map_location=pl_module.device, weights_only=False)
    pl_module.load_state_dict(ckpt["state_dict"])
    return best_path


class Stage1RecCheck(Callback):
    """Stage 1 reconstruction check: encoder -> VQ -> decoder roundtrip with
    ground-truth atom ordering, on up to ``n_samples`` val molecules. Logs
    ``stage1/<metric>`` and writes ``stage1_rec_check.json`` next to the run
    logs.
    """

    def __init__(
        self, n_samples: int = 5000, output_filename: str = "stage1_rec_check.json"
    ):
        super().__init__()
        self.n_samples = int(n_samples)
        self.output_filename = output_filename

    def on_train_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        loader = fresh_val_loader(trainer, pl_module.batch_size, pyg=True)
        if loader is None:
            return

        loaded_ckpt = load_best_ckpt_into(trainer, pl_module)

        was_training = pl_module.training
        pl_module.eval()
        running: dict = {}
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(pl_module.device)
                preds = stage1_forward(pl_module, batch)
                batch_totals = per_mol_metrics(*preds)
                for k, v in batch_totals.items():
                    running[k] = running.get(k, 0) + v
                if running.get("n_mols", 0) >= self.n_samples:
                    break
        if was_training:
            pl_module.train()

        rates = to_rates(running)
        rates["stage"] = "stage1_vqgae_roundtrip"
        rates["ckpt"] = loaded_ckpt
        log_gate_metrics(trainer, "stage1", rates)
        out_path = resolve_log_dir(trainer) / self.output_filename
        out_path.write_text(json.dumps(rates, indent=2))
        print(f"[Stage1RecCheck] {rates}")


class Stage2RecCheck(Callback):
    """Stage 2 reconstruction check: ONN reorders codebook indices, then a
    fixed VQGAE decodes both the ground-truth ordering and the ONN-canonical
    ordering. We compare the resulting chython graphs. The VQGAE checkpoint
    must match the one that produced the ONN training data, otherwise the
    codebook semantics mismatch and the comparison is meaningless.
    """

    def __init__(
        self,
        vqgae_ckpt: str,
        n_samples: int = 5000,
        output_filename: str = "stage2_rec_check.json",
    ):
        super().__init__()
        self.vqgae_ckpt = vqgae_ckpt
        self.n_samples = int(n_samples)
        self.output_filename = output_filename

    def on_train_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        loader = fresh_val_loader(trainer, pl_module.batch_size, pyg=False)
        if loader is None:
            return

        from .models import VQGAE
        from .utils import restore_order

        device = pl_module.device
        vqgae = VQGAE.load_from_checkpoint(self.vqgae_ckpt, map_location=device)
        vqgae.task = "decode"
        vqgae.eval().to(device)

        loaded_onn_ckpt = load_best_ckpt_into(trainer, pl_module)

        was_training = pl_module.training
        pl_module.eval()

        running = {
            "n_mols": 0,
            "exact_order_match": 0,
            "decoded_valid_gt": 0,
            "decoded_valid_pred": 0,
            "graph_match": 0,
        }
        with torch.no_grad():
            for batch in loader:
                gt_inds = batch[0].to(device).long()
                canon_inds, _ = restore_order(gt_inds, pl_module)
                canon_inds = canon_inds.to(device)

                gt_atoms, gt_bonds, gt_sizes = vqgae([gt_inds])
                pr_atoms, pr_bonds, _ = vqgae([canon_inds])

                gt_inds_np = gt_inds.cpu().numpy()
                canon_np = canon_inds.cpu().numpy()
                gt_atoms_np = gt_atoms.cpu().numpy()
                gt_bonds_np = gt_bonds.cpu().numpy()
                gt_sizes_np = gt_sizes.cpu().numpy()
                pr_atoms_np = pr_atoms.cpu().numpy()
                pr_bonds_np = pr_bonds.cpu().numpy()

                for i in range(gt_inds_np.shape[0]):
                    size = int(gt_sizes_np[i])
                    if size < 2:
                        continue
                    running["n_mols"] += 1
                    if (canon_np[i, :size] == gt_inds_np[i, :size]).all():
                        running["exact_order_match"] += 1
                    mol_gt = decoded_mol(
                        gt_atoms_np[i], gt_bonds_np[i], size, canonicalize=True
                    )
                    mol_pr = decoded_mol(
                        pr_atoms_np[i], pr_bonds_np[i], size, canonicalize=True
                    )
                    if mol_gt is not None:
                        running["decoded_valid_gt"] += 1
                    if mol_pr is not None:
                        running["decoded_valid_pred"] += 1
                    if mol_gt is not None and mol_pr is not None and mol_gt == mol_pr:
                        running["graph_match"] += 1
                if running["n_mols"] >= self.n_samples:
                    break
        if was_training:
            pl_module.train()

        n = max(running["n_mols"], 1)
        rates = {
            "stage": "stage2_onn_with_fixed_vqgae",
            "vqgae_ckpt": self.vqgae_ckpt,
            "onn_ckpt": loaded_onn_ckpt,
            "n_mols": running["n_mols"],
            "exact_order_match": running["exact_order_match"] / n,
            "decoded_valid_gt": running["decoded_valid_gt"] / n,
            "decoded_valid_pred": running["decoded_valid_pred"] / n,
            "graph_match": running["graph_match"] / n,
        }
        log_gate_metrics(trainer, "stage2", rates)
        out_path = resolve_log_dir(trainer) / self.output_filename
        out_path.write_text(json.dumps(rates, indent=2))
        print(f"[Stage2RecCheck] {rates}")
