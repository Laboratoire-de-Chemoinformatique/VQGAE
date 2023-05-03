import click
import torch
import yaml
from collections import defaultdict
from CGRtools.containers import MoleculeContainer
from torch_geometric.utils import degree

training_config = {
    "seed_everything": 42,
    "trainer": {
        "logger": {
            "class_path": "pytorch_lightning.loggers.CSVLogger",
            "init_args": {
                "save_dir": "/home/.../logs",
                "name": "vqgae_default",
                "version": 1,
            }
        },
        "gradient_clip_val": 1.0,
        "gpus": 0,
        "max_epochs": 100,
        "log_every_n_steps": 1000,
        "precision": 16,
        "detect_anomaly": False,
    },
    "model": {
        "max_atoms": 51,
        "batch_size": 500,
        "num_conv_layers": 5,
        "vector_dim": 512,
        "num_mha_layers": 8,
        "num_agg_layers": 2,
        "num_heads_encoder": 16,
        "num_heads_decoder": 16,
        "dropout": 0.2,
        "vq_embeddings": 4096,
        "bias": True,
        "init_values": 0.0001,
        "lr": 0.0002,
        "task": "train",
        "shuffle_graph": False,
        "positional_bias": False,
        "reparam": False,
        "class_categories": [38, 29, 21, 25, 16, 13, 50, 13]
    },
    "data": {
        "path_train_predict": "/home/.../chembl_train.sdf",
        "path_val": "/home/.../chembl_val.sdf",
        "tmp_folder": "/home/.../tmp/",
        "tmp_name": "chembl",
        "num_workers": 0,
        "pin_memory": True,
        "drop_last": True,
        "properties_names": {
            "Heavy Atom Count": "class",
            "Hetero Atom Count": "class",
            "acceptorcount": "class",
            "donorcount": "class",
            "Chiral center count": "class",
            "Ring count": "class",
            "Hetero ring count": "class",
            "Rotatable bond count": "class",
            "Aromatic ring count": "class"
        }
    },
    "vqgae_lr_monitor": {
        "logging_interval": "epoch",
    },
    "vqgae_model_checkpoint": {
        "dirpath": "/home/.../weights",
        "filename": "${trainer.logger.init_args.name}",
        "monitor": "train_loss",
        "mode": "min"
    }
}


@click.command()
@click.option(
    "--task",
    required=True,
    help="For which task generate default config. Options are: train, encode, decode",
)
def vqgae_default_config(task):
    with open(f"default_vqgae_config_{task}.yaml", "w") as file:
        if task == 'train':
            yaml.dump(training_config, file)
        elif task == 'encode':
            yaml.dump(training_config, file)
        elif task == 'decode':
            yaml.dump(training_config, file)
        else:
            raise ValueError(f"I don't know this task: {task}")


def create_chem_graph(atoms_sequence, connectivity_matrix, size, atom_types) -> MoleculeContainer:
    """
    Create molecular graph or basis for HLG
    :param size: size of molecule
    :param atoms_sequence: sequence of atoms
    :param atom_types: atoms types
    :param connectivity_matrix: adjacency matrix
    :return: molecular graph
    """
    molecule = MoleculeContainer()
    for n in range(size):
        atom = atoms_sequence[n]
        atomic_symbol, charge = atom_types[int(atom)]
        molecule.add_atom(atom=atomic_symbol, charge=charge)

    for i in range(len(molecule)):
        for j in range(i + 1, len(molecule)):
            if connectivity_matrix[i][j]:
                molecule.add_bond(i + 1, j + 1, int(connectivity_matrix[i][j]))

    return molecule


def get_mol_names(sdf_file):
    """
    Given a file, return a tuple of the names of the molecules in the file.

    :param sdf_file: the file to read from
    :return: A tuple of the names of the molecules in the SDF file.
    """
    with open(sdf_file, 'r') as inp:
        line = inp.read()
    names = [mol.split('\n')[0] for mol in line.split('$$$$\n') if mol]
    del line
    return tuple(names)


def write_svm(index, vector, file):
    num_columns = vector.shape[0] - 1
    file.write(str(index))
    for j in range(num_columns):
        if vector[j]:
            file.write(f' {j + 1}:{vector[j]}')
    file.write(f' {num_columns + 1}:{vector[num_columns]}\n')


def write_svm_cb(index, codebook_indices, mol_size, file):
    file.write(str(index))
    for i, val in codebook_indices.items():
        file.write(f' {i + 1}:{val + 1}')  # TODO: In decoder not to forget that these indices with + 1
    if mol_size - 1 not in codebook_indices.keys():
        file.write(f' {mol_size}:0\n')


def pna_deg_stats(dataset):
    deg = torch.zeros(7, dtype=torch.long)
    for data, _, _ in dataset:
        d = degree(data.edge_index[1].long(), num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    return deg


def convert_codebook_indices(cb_vecs, mol_sizes):
    canon_order_inds = []
    for i in range(cb_vecs.shape[0]):
        vec = cb_vecs[i, :int(mol_sizes[i])]
        ordered_tmp = defaultdict(int)
        for j in range(vec.shape[-1]):
            ordered_tmp[j] = int(vec[j])
        canon_order_inds.append(ordered_tmp)
    return canon_order_inds
