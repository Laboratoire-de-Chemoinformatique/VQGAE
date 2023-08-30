import torch
from abc import ABC
from pathlib import Path

from CGRtools.files import SDFWrite
from pytorch_lightning.callbacks import BasePredictionWriter
from safetensors.torch import save_file

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
            "features": torch.reshape(torch.stack(self.features_tmp), (-1, feature_dim)),
            "codebook": torch.reshape(torch.stack(self.codebook_tmp), (-1, codebook_dim))
        }
        if self.use_chunks:
            output_file = self.output_dir.joinpath(f"{self.output_name}_chunk_{self.chunk_id:03}.safetensors")
        else:
            output_file = self.output_dir.joinpath(f"{self.output_name}.safetensors")
        save_file(output, str(output_file))
        if self.use_chunks:
            del self.features_tmp
            del self.codebook_tmp
            self.features_tmp = []
            self.codebook_tmp = []

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
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
