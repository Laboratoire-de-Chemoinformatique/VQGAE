import heapq
from typing import Dict

import click
import numpy as np
import torch
import yaml
from CGRtools.containers import MoleculeContainer
from CGRtools.exceptions import InvalidAromaticRing
from scipy.optimize import linear_sum_assignment

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


def morgan(mol: MoleculeContainer) -> Dict[int, float]:
    # Initialize atom values
    atom_vals = {}
    for idx, atom in mol._atoms.items():
        atom_num = atom.atomic_number * 2 - sum(int(b) for b in mol._bonds[idx].values()) - atom.charge
        if atom.in_ring:
            atom_num += 0.5
        atom_vals[idx] = atom_num

    limit = len(mol._atoms) - 1
    prev_count = 0
    stab_count = 0

    for _ in range(limit):
        # Update atom values based on neighbors
        new_vals = {}
        for idx, atom in mol._atoms.items():
            new_val = atom_vals[idx] + sum(atom_vals[n] for n in mol._bonds[idx])
            new_vals[idx] = new_val

        atom_vals = new_vals

        # Check for uniqueness
        uniq_count = len(set(atom_vals.values()))
        if uniq_count == len(atom_vals):  # each atom now unique
            break
        elif uniq_count == prev_count:  # not changed. molecules like benzene
            if stab_count == 3:
                break
            stab_count += 1
        elif stab_count:  # changed unique atoms number. reset stability check.
            stab_count = 0

        prev_count = uniq_count

    return atom_vals


def frag_inds_to_counts(raw_vectors, num_frags=4096):
    counts_vec = np.zeros((raw_vectors.shape[0], num_frags), dtype=np.uint8)
    for i in range(raw_vectors.shape[0]):
        for j in range(raw_vectors.shape[1]):
            cb_indx = raw_vectors[i, j]
            if cb_indx > -1:
                counts_vec[i, cb_indx] += 1
            else:
                break
    return counts_vec


def frag_counts_to_inds(frag_counts: np.ndarray, max_atoms=51):
    frag_inds = -1 * torch.ones((frag_counts.shape[0], max_atoms), dtype=torch.int64)
    for mol_id in range(frag_counts.shape[0]):
        atom_counter = 0
        for frag_id in range(frag_counts.shape[1]):
            frag_count = frag_counts[mol_id, frag_id]
            if frag_count:
                for _ in range(frag_count):
                    frag_inds[mol_id, atom_counter] = frag_id
                    atom_counter += 1
                    if atom_counter == max_atoms:
                        break
                if atom_counter == max_atoms:
                    break
    return frag_inds


def find_best_permutation(perm_matrix: np.ndarray):
    num_rows, num_cols = np.shape(perm_matrix)
    if num_rows != num_cols:
        raise ValueError("The permutation matrix must be square.")

    original_cost_matrix = -np.array(perm_matrix)
    row_ind, col_ind = linear_sum_assignment(original_cost_matrix)
    score = -original_cost_matrix[row_ind, col_ind].mean()
    reorder_indices = np.argsort(col_ind)
    return reorder_indices, score


def restore_order(frag_inds, ordering_model):
    scores = []

    canon_order_inds = -1 * torch.ones_like(frag_inds)

    mol_sizes = torch.where(frag_inds > -1, 1, 0).sum(-1)
    inputs_order_inds, _ = torch.sort(frag_inds, descending=True)

    with torch.no_grad():
        results = ordering_model([frag_inds])

    for mol_i in range(frag_inds.shape[0]):
        mol_size = mol_sizes[mol_i]
        input_order_inds = inputs_order_inds[mol_i, :mol_size]
        result_mol = results[mol_i, :mol_size, :mol_size].numpy()
        top_solution, score = find_best_permutation(result_mol)
        canon_order_inds[mol_i, :mol_size] = input_order_inds[top_solution]
        scores.append(score)

    return canon_order_inds, scores


def decode_molecules(ordered_frag_inds, vqgae_model):
    decoded_molecules = []
    validity = []
    prediction = vqgae_model([ordered_frag_inds])

    for mol_i in range(prediction[0].shape[0]):
        molecule = create_chem_graph(
            prediction[0][mol_i],
            prediction[1][mol_i],
            int(prediction[2][mol_i]),
        )
        molecule.clean2d()
        valid = False
        if molecule.connected_components_count == 1:
            if not molecule.check_valence():
                try:
                    molecule.canonicalize()
                except InvalidAromaticRing:
                    continue
                valid = True
        decoded_molecules.append(molecule)
        validity.append(valid)
    return decoded_molecules, validity
