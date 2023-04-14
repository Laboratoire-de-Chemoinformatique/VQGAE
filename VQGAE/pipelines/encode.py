from pytorch_lightning.cli import LightningCLI

from ..callbacks import EncoderPredictionsWriter
from ..models import VQGAE
from ..preprocessing import VQGAEData


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


def encoding_interface():
    cli = EncodingCLI(
        model_class=VQGAE,
        datamodule_class=VQGAEData,
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_overwrite=True,
        run=False
    )
    if cli.config.ckpt_model_file is None:
        raise ValueError("Model path is not specified in ckpt_model_file parameter")
    cli.trainer.predict(cli.model, cli.datamodule, return_predictions=False, ckpt_path=cli.config.ckpt_model_file)


if __name__ == "__main__":
    encoding_interface()
