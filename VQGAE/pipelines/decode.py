from pytorch_lightning.cli import LightningCLI

from ..callbacks import DecoderPredictionsWriter
from ..models import VQGAE
from ..preprocessing import VQGAEVectors


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


def decoding_interface():
    cli = DecodingCLI(
        model_class=VQGAE,
        datamodule_class=VQGAEVectors,
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_overwrite=True,
        run=False
    )
    if cli.config.ckpt_model_file is None:
        raise ValueError("Model path is not specified in ckpt_model_file parameter")
    cli.trainer.predict(cli.model, cli.datamodule, return_predictions=False, ckpt_path=cli.config.ckpt_model_file)


if __name__ == "__main__":
    decoding_interface()
