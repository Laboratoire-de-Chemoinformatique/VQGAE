import click
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.cli import LightningCLI

from .callbacks import (
    DecoderPredictionsWriter,
    EncoderPredictionsWriter,
    Stage1RecCheck,
    Stage2RecCheck,
)
from .models import VQGAE, OrderingNetwork
from .preprocessing import VQGAEData, VQGAEVectors


class TrainingCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(LearningRateMonitor, "vqgae_lr_monitor")
        parser.add_lightning_class_args(ModelCheckpoint, "vqgae_model_checkpoint")
        parser.add_lightning_class_args(Stage1RecCheck, "stage1_rec_check")
        parser.set_defaults(
            {
                "model.task": "train",
                "vqgae_lr_monitor.logging_interval": "epoch",
                "vqgae_model_checkpoint.mode": "min",
                "vqgae_model_checkpoint.monitor": "train_loss",
            }
        )
        parser.link_arguments("model.max_atoms", "data.max_atoms")
        parser.link_arguments("model.batch_size", "data.batch_size")

    # currently the compiled model does not work properly
    # def fit(self, model, **kwargs):
    #     compiled_model = torch.compile(model, mode="reduce-overhead")
    #     self.trainer.fit(compiled_model, **kwargs)


class EncodingCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--ckpt_model_file")
        parser.add_lightning_class_args(EncoderPredictionsWriter, "encoder_writer")
        parser.set_defaults({"model.task": "encode"})
        parser.link_arguments("model.max_atoms", "data.max_atoms")
        parser.link_arguments("model.batch_size", "data.batch_size")


class DecodingCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--ckpt_model_file")
        parser.add_lightning_class_args(DecoderPredictionsWriter, "decoder_writer")
        parser.set_defaults({"model.task": "decode"})
        parser.link_arguments("model.batch_size", "data.batch_size")


class TrainingONNCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(
            LearningRateMonitor, "ordering_network_lr_monitor"
        )
        parser.add_lightning_class_args(
            ModelCheckpoint, "ordering_network_model_checkpoint"
        )
        parser.add_lightning_class_args(Stage2RecCheck, "stage2_rec_check")
        parser.set_defaults(
            {
                "ordering_network_lr_monitor.logging_interval": "epoch",
                "ordering_network_model_checkpoint.mode": "min",
                "ordering_network_model_checkpoint.monitor": "train_loss",
            }
        )
        parser.link_arguments("model.batch_size", "data.batch_size")


def training_interface():
    TrainingCLI(
        model_class=VQGAE,
        datamodule_class=VQGAEData,
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_kwargs={"overwrite": True},
    )


def encoding_interface():
    cli = EncodingCLI(
        model_class=VQGAE,
        datamodule_class=VQGAEData,
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_kwargs={"overwrite": True},
        run=False,
    )
    if cli.config.ckpt_model_file is None:
        raise ValueError("Model path is not specified in ckpt_model_file parameter")
    cli.trainer.predict(
        cli.model,
        cli.datamodule,
        return_predictions=False,
        ckpt_path=cli.config.ckpt_model_file,
    )


def decoding_interface():
    cli = DecodingCLI(
        model_class=VQGAE,
        datamodule_class=VQGAEVectors,
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_kwargs={"overwrite": True},
        run=False,
    )
    if cli.config.ckpt_model_file is None:
        raise ValueError("Model path is not specified in ckpt_model_file parameter")
    cli.trainer.predict(
        cli.model,
        cli.datamodule,
        return_predictions=False,
        ckpt_path=cli.config.ckpt_model_file,
    )


def ordering_interface():
    TrainingONNCLI(
        model_class=OrderingNetwork,
        datamodule_class=VQGAEVectors,
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_kwargs={"overwrite": True},
    )


# ---------------------------------------------------------------------------
# vqgae_preprocess_shards — high-throughput SMILES/SDF -> safetensors shards
# ---------------------------------------------------------------------------

# The eight ChEMBL meta keys the legacy SDF training pipeline expects. Used
# only when --props sdf-meta is selected; --props rdkit/none ignore them.
_DEFAULT_SDF_META_KEYS = (
    "Hetero Atom Count",
    "acceptorcount",
    "donorcount",
    "Chiral center count",
    "Ring count",
    "Hetero ring count",
    "Rotatable bond count",
    "Aromatic ring count",
)


@click.command(name="vqgae_preprocess_shards")
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Input file: .smi[.gz], .csv[.gz], or .sdf[.gz].",
)
@click.option(
    "-o",
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Where shard_NNNNN.safetensors files will be written.",
)
@click.option(
    "--shard-size",
    default=100_000,
    show_default=True,
    type=int,
    help="Number of molecules per shard.",
)
@click.option(
    "--max-atoms",
    default=51,
    show_default=True,
    type=int,
    help="Skip molecules with more heavy atoms than this.",
)
@click.option(
    "--n-jobs",
    default=-1,
    show_default=True,
    type=int,
    help="Joblib worker count for parallel preprocessing.",
)
@click.option(
    "--props",
    "props_mode",
    type=click.Choice(["rdkit", "sdf-meta", "none"]),
    default=None,
    help=(
        "How to populate the multi-target classifier labels per molecule. "
        "Default: 'sdf-meta' for .sdf inputs, 'rdkit' otherwise."
    ),
)
@click.option(
    "--smiles-column",
    default="0",
    show_default=True,
    help="For .csv: column header (string) or 0-based index (int).",
)
@click.option(
    "--id-column",
    default=None,
    help="Optional ID column name/index for .csv inputs.",
)
@click.option(
    "--start-at-shard",
    default=0,
    show_default=True,
    type=int,
    help="Resume aborted preprocessing at this shard index.",
)
@click.option(
    "--chunk-size",
    default=5000,
    show_default=True,
    type=int,
    help="Per-batch parallel chunk size (lower = lower peak RAM).",
)
def preprocess_shards_interface(
    input_path: str,
    output_dir: str,
    shard_size: int,
    max_atoms: int,
    n_jobs: int,
    props_mode: str | None,
    smiles_column: str,
    id_column: str | None,
    start_at_shard: int,
    chunk_size: int,
) -> None:
    """High-throughput SMILES/SDF -> safetensors shards preprocessor.

    Pipeline: stream -> rdkit parse -> joblib parallel preprocess ->
    pack into N=SHARD_SIZE molecule shards. Writes an `index.json` next
    to the shards summarising per-shard counts (used by ShardedVQGAEDataset).

    Examples:

        vqgae_preprocess_shards -i enamine_real.smi.gz -o ./shards/

        vqgae_preprocess_shards -i actives.csv -o ./shards/ \\
            --smiles-column smiles --props rdkit

        vqgae_preprocess_shards -i chembl_train.sdf -o ./shards/train \\
            --props sdf-meta
    """
    from .preprocessing import preprocess_to_shards

    # Decide default props_mode by suffix.
    name = input_path.lower()
    if props_mode is None:
        props_mode = "sdf-meta" if ".sdf" in name else "rdkit"

    # Coerce smiles_column to int when it parses as one.
    try:
        smi_col: str | int = int(smiles_column)
    except ValueError:
        smi_col = smiles_column

    id_col: str | int | None = None
    if id_column is not None:
        try:
            id_col = int(id_column)
        except ValueError:
            id_col = id_column

    preprocess_to_shards(
        input_path,
        output_dir,
        shard_size=shard_size,
        max_atoms=max_atoms,
        n_jobs=n_jobs,
        props_mode=props_mode,
        sdf_meta_keys=_DEFAULT_SDF_META_KEYS if props_mode == "sdf-meta" else None,
        smiles_column=smi_col,
        id_column=id_col,
        start_at_shard=start_at_shard,
        chunk_size=chunk_size,
    )


# ---------------------------------------------------------------------------
# vqgae_train_sharded — same as vqgae_train fit but datamodule is sharded
# ---------------------------------------------------------------------------


def training_sharded_interface():
    """Like ``vqgae_train fit`` but uses :class:`ShardedVQGAEData`.

    Config YAML's ``data:`` block must match :class:`ShardedVQGAEData`'s
    init kwargs (``train_dir``, ``val_dir``, ``batch_size``, ``max_atoms``,
    ``num_workers``, ``pin_memory``, ``stateful``, ...).
    """
    from .preprocessing import ShardedVQGAEData

    TrainingCLI(
        model_class=VQGAE,
        datamodule_class=ShardedVQGAEData,
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_kwargs={"overwrite": True},
    )
