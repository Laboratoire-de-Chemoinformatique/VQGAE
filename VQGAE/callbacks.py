import pickle
from abc import ABC
from pathlib import Path

from CGRtools.files import SDFWrite
from pytorch_lightning.callbacks import BasePredictionWriter
from tqdm import tqdm

from .utils import create_chem_graph, write_svm, get_mol_names


class EncoderPredictionsWriter(BasePredictionWriter, ABC):
    def __init__(
            self,
            input_file: str,
            output_dir: str,
            codebook_dir: str,
            use_names=False,
            names_file="tmp/tmp_names.pickle",
            chunk_size=400,
    ):
        super().__init__()
        self.i = 0
        self.output_file = open(output_dir, "w")
        self.codebook_file = open(codebook_dir, "w")
        self.chunk_size = chunk_size
        self.tmp = []
        self.use_names = use_names
        if self.use_names:
            if names_file:
                if Path(names_file).is_file():
                    with open(names_file, "rb") as inp:
                        self.names = pickle.load(inp)
                else:
                    self.names = get_mol_names(input_file)
                    with open(names_file, "wb") as out:
                        pickle.dump(self.names, out)
            else:
                self.names = get_mol_names(input_file)

    def write_file(self, prediction):
        for j in range(prediction[0].shape[0]):
            if self.use_names:
                if self.i < len(self.names):
                    name = self.names[self.i]
                    write_svm(name, prediction[0][j], self.output_file)
                    write_svm(name, prediction[1][j], self.codebook_file)
                else:
                    break
            else:
                write_svm(self.i, prediction[0][j], self.output_file)
                write_svm(self.i, prediction[1][j], self.codebook_file)
            self.output_file.flush()
            self.codebook_file.flush()
            self.i += 1

    def write_chunk(self):
        for prediction in tqdm(self.tmp):
            self.write_file(prediction)

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        encoder_prediction = (prediction[1].cpu().numpy(), prediction[2].cpu().numpy(),)
        if self.chunk_size > 1:
            self.tmp.append(encoder_prediction)
            if len(self.tmp) % self.chunk_size == 0:
                self.write_chunk()
                del self.tmp
                self.tmp = []
        else:
            self.write_file(encoder_prediction)

    def on_predict_epoch_end(self, trainer, pl_module, outputs):
        if self.tmp:
            self.write_chunk()
        self.output_file.close()
        self.codebook_file.close()


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
                self.get_atoms_types,
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

    def on_predict_epoch_end(self, trainer, pl_module, outputs):
        self.output_file.close()

    @property
    def get_atoms_types(self):
        return (("C", 0), ("S", 0), ("Se", 0), ("F", 0), ("Cl", 0), ("Br", 0), ("I", 0),
                ("B", 0), ("P", 0), ("Si", 0), ("O", 0), ("O", -1), ("N", 0), ("N", 1), ("N", -1),)
