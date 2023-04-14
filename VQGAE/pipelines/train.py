from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.cli import LightningCLI

from ..models import VQGAE
from ..preprocessing import VQGAEData


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


def training_interface():
    cli = TrainingCLI(
        model_class=VQGAE,
        datamodule_class=VQGAEData,
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_overwrite=True,
    )


if __name__ == "__main__":
    training_interface()
