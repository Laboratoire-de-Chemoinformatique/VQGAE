from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.cli import LightningCLI

from .models import VQGAE
from .preprocessing import VQGAEData, VQGAEVectors
from .callbacks import EncoderPredictionsWriter, DecoderPredictionsWriter


class TrainingCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(LearningRateMonitor, "vqgae_lr_monitor")
        parser.add_lightning_class_args(ModelCheckpoint, "vqgae_model_checkpoint")
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


class EncodingCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--ckpt_model_file")
        parser.add_lightning_class_args(EncoderPredictionsWriter, "encoder_writer")
        parser.set_defaults(
            {
                "model.task": "encode"
            }
        )
        parser.link_arguments("model.max_atoms", "data.max_atoms")
        parser.link_arguments("model.batch_size", "data.batch_size")
        parser.link_arguments("data.path_train_predict", "encoder_writer.input_file")


class DecodingCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--ckpt_model_file")
        parser.add_lightning_class_args(DecoderPredictionsWriter, "decoder_writer")
        parser.set_defaults(
            {
                "model.task": "decode"
            }
        )
        parser.link_arguments("model.max_atoms", "data.max_atoms")
        parser.link_arguments("model.batch_size", "data.batch_size")


def training_interface():
    TrainingCLI(
        model_class=VQGAE,
        datamodule_class=VQGAEData,
        parser_kwargs={"parser_mode": "omegaconf"},
    )


def encoding_interface():
    cli = EncodingCLI(
        model_class=VQGAE,
        datamodule_class=VQGAEData,
        parser_kwargs={"parser_mode": "omegaconf"},
        run=False
    )
    if cli.config.ckpt_model_file is None:
        raise ValueError("Model path is not specified in ckpt_model_file parameter")
    cli.trainer.predict(cli.model, cli.datamodule, return_predictions=False, ckpt_path=cli.config.ckpt_model_file)


def decoding_interface():
    cli = DecodingCLI(
        model_class=VQGAE,
        datamodule_class=VQGAEVectors,
        parser_kwargs={"parser_mode": "omegaconf"},
        run=False
    )
    if cli.config.ckpt_model_file is None:
        raise ValueError("Model path is not specified in ckpt_model_file parameter")
    cli.trainer.predict(cli.model, cli.datamodule, return_predictions=False, ckpt_path=cli.config.ckpt_model_file)
