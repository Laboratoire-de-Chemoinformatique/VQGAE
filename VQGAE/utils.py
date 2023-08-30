import heapq

from typing import Dict

import click
import yaml
from CGRtools.containers import MoleculeContainer

accepted_atoms = ('C', 'N', 'S', 'O', 'Se', 'F', 'Cl', 'Br', 'I', 'B', 'P', 'Si')

atoms_types = (('C', 0), ('S', 0), ('Se', 0), ('F', 0), ('Cl', 0), ('Br', 0),
               ('I', 0), ('B', 0), ('P', 0), ('Si', 0), ('O', 0), ('O', -1), ('N', 0), ('N', 1), ('N', -1))

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


def create_chem_graph(atoms_sequence, connectivity_matrix, size) -> MoleculeContainer:
    """
    Create molecular graph or basis for HLG
    :param size: size of molecule
    :param atoms_sequence: sequence of atoms
    :param connectivity_matrix: adjacency matrix
    :return: molecular graph
    """
    molecule = MoleculeContainer()
    for n in range(size):
        atom = atoms_sequence[n]
        atomic_symbol, charge = atoms_types[int(atom)]  # defined on the line 9
        molecule.add_atom(atom=atomic_symbol, charge=charge)

    for i in range(len(molecule)):
        for j in range(i + 1, len(molecule)):
            if connectivity_matrix[i][j]:
                molecule.add_bond(i + 1, j + 1, int(connectivity_matrix[i][j]))

    return molecule


def beam_search(matrix, beam_width=5):
    # Initialize priority queue with an empty permutation and mean probability 0
    pq = [(-0.0, [])]
    best_perms = []

    for col in range(len(matrix[0])):
        new_pq = []

        # For each partial permutation in the priority queue
        for mean_prob, partial_perm in pq:
            for row in range(len(matrix)):
                if row not in partial_perm:
                    # Calculate new mean probability
                    new_mean_prob = ((-mean_prob * len(partial_perm)) + matrix[row][col]) / (len(partial_perm) + 1)

                    # Add new partial permutation to the new priority queue
                    new_partial_perm = partial_perm + [row]
                    heapq.heappush(new_pq, (-new_mean_prob, new_partial_perm))

                    # If the new partial permutation is complete, add it to the best_perms list
                    if len(new_partial_perm) == len(matrix[0]):
                        heapq.heappush(best_perms, (-new_mean_prob, new_partial_perm))

        # Keep only the top-k = beam_width partial permutations
        pq = heapq.nsmallest(beam_width, new_pq)

    # Get all complete permutations
    top_perms = heapq.nsmallest(beam_width, best_perms)
    top_perms = [(perm, -mean_prob) for mean_prob, perm in top_perms]
    return top_perms


def _morgan(atoms: Dict[int, int], bonds: Dict[int, Dict[int, int]]) -> Dict[int, int]:
    """
    Adopted from https://github.com/chython/chython/blob/master/chython/algorithms/morgan.py
    :param atoms: hashes of atoms
    :param bonds: hashes of bonds
    :return: unique hashes of atoms after Morgan algorithm
    """
    tries = len(atoms) - 1
    numb = len(set(atoms.values()))
    stab = 0

    for _ in range(tries):
        atoms = {n: hash((atoms[n], *(x for x in sorted((atoms[m], b) for m, b in ms.items()) for x in x)))
                 for n, ms in bonds.items()}
        old_numb, numb = numb, len(set(atoms.values()))
        if numb == len(atoms):  # each atom now unique
            break
        elif numb == old_numb:  # not changed. molecules like benzene
            if stab == 3:
                break
            stab += 1
        elif stab:  # changed unique atoms number. reset stability check.
            stab = 0

    return atoms