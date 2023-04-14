import os.path as osp
from abc import ABC

import numpy as np
import torch
from CGRtools.containers import MoleculeContainer
from CGRtools.files import SDFRead
from mendeleev import element
from pytorch_lightning import LightningDataModule
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.lightning import LightningDataset
from torch_geometric.loader.dataloader import DataLoader as PYGDataLoader
from torch_geometric.transforms import ToUndirected
from tqdm import tqdm


def calc_atoms_info(accepted_atoms: tuple) -> dict:
    """
    Given a tuple of accepted atoms, return a dictionary with the atom symbol as the key and a tuple of
    the period, group, subshell, and number of electrons as the value.

    :param accepted_atoms: tuple of strings
    :type accepted_atoms: tuple
    :return: A dictionary with the atomic number as the key and the period, group, shell, and number of
    electrons as the value.
    """
    mendel_info = {}
    shell_to_num = {'s': 1, 'p': 2, 'd': 3, 'f': 4}
    for atom in accepted_atoms:
        mendel_atom = element(atom)
        period = mendel_atom.period
        group = mendel_atom.group_id
        shell, electrons = mendel_atom.ec.last_subshell()
        mendel_info[atom] = (period, group, shell_to_num[shell[1]], electrons)
    return mendel_info


def atom_to_vector(atom, mendel_info: dict):
    """
    Given an atom, return a vector of length 8 with the following information:

    1. Atomic number
    2. Period
    3. Group
    4. Number of electrons + atom's charge
    5. Shell
    6. Total number of hydrogens
    7. Whether the atom is in a ring
    8. Number of neighbors

    :param atom: the atom object
    :param mendel_info: a dictionary of the form {'C': (3, 1, 1, 2), 'O': (3, 1, 1, 2), ...}
    :type mendel_info: dict
    :return: The vector of the atom.
    """
    vector = np.zeros(8, dtype=np.int8)
    period, group, shell, electrons = mendel_info[atom.atomic_symbol]
    vector[0] = atom.atomic_number
    vector[1] = period
    vector[2] = group
    vector[3] = electrons + atom.charge
    vector[4] = shell
    vector[5] = atom.total_hydrogens
    vector[6] = int(atom.in_ring)
    vector[7] = atom.neighbors
    return vector


def bonds_to_vector(molecule: MoleculeContainer, atom_ind: int):
    vector = np.zeros(3, dtype=np.int8)
    for b_order in molecule._bonds[atom_ind].values():
        vector[int(b_order) - 1] += 1
    return vector


def graph_to_atoms_vectors(molecule: MoleculeContainer, max_atoms: int, mendel_info: dict):
    """
    Given a molecule, it returns a vector of shape (max_atoms, 12) where each row is an atom and each
    column is a feature.

    :param molecule: The molecule to be converted to a vector
    :type molecule: MoleculeContainer
    :param max_atoms: The maximum number of atoms in the molecule
    :type max_atoms: int
    :param mendel_info: a dictionary containing the information about the Mendel system
    :type mendel_info: dict
    :return: The atoms_vectors array
    """
    atoms_vectors = np.zeros((max_atoms, 11), dtype=np.int8)
    for n, atom in molecule.atoms():
        atoms_vectors[n - 1][:8] = atom_to_vector(atom, mendel_info)
    for n, _ in molecule.atoms():
        atoms_vectors[n - 1][8:] = bonds_to_vector(molecule, n)

    return atoms_vectors


def graph_to_bond_matrix(molecule: MoleculeContainer, max_atoms: int):
    """
    it takes a molecule and returns a bond matrix.

    :param molecule: The molecule to be converted
    :type molecule: MoleculeContainer
    :param max_atoms: The maximum number of atoms in the molecule
    :type max_atoms: int
    :return: The bond matrix.
    """
    bond_matrix = np.zeros((max_atoms, max_atoms), dtype=np.int8)

    for a, n, order in molecule.bonds():
        bond_matrix[a - 1][n - 1] = bond_matrix[n - 1][a - 1] = int(order)
        if int(order) == 4:
            raise ValueError('Found a structure with aromatic bond')

    return bond_matrix


def generate_true_vector(molecule: MoleculeContainer, max_atoms: int, atoms_types: tuple):
    """
    It takes a molecule, the maximum number of atoms in the molecule, and the list of atoms types. It
    then generates a numpy array of shape (max_atoms, 1) with the atom types of the atoms in the
    molecule.
    
    :param molecule: The molecule to be used for training
    :param max_atoms: the maximum number of atoms in the molecule
    :param atoms_types: tuple of tuples, each tuple contains the atomic symbol and the charge of the
    atom
    :return: A vector of integers, where each integer is the index of the atom type in the atoms_types
    tuple.
    """
    y_true = -1 * np.ones(max_atoms, dtype=np.int8)

    for n, atom in molecule.atoms():
        y_true[n - 1] = atoms_types.index((atom.atomic_symbol, atom.charge))
    return y_true


def generate_true_vector_2(molecule: MoleculeContainer, atoms_types: tuple):
    """
    It takes a molecule, the maximum number of atoms in the molecule, and the list of atoms types. It
    then generates a numpy array of shape (max_atoms, 1) with the atom types of the atoms in the
    molecule.

    :param molecule: The molecule to be used for training
    :param max_atoms: the maximum number of atoms in the molecule
    :param atoms_types: tuple of tuples, each tuple contains the atomic symbol and the charge of the
    atom
    :return: A vector of integers, where each integer is the index of the atom type in the atoms_types
    tuple.
    """
    y_true = []
    for n, atom in molecule.atoms():
        y_true.append(atoms_types.index((atom.atomic_symbol, atom.charge)))
    return y_true


def preprocess_molecules(file, max_atoms, properties_file=None, meta_name_y=None):
    """
    Given a molecule, it generates a pytorch geometric graph object, a one-hot encoded vector of the
    atoms, and a matrix of the bonds.
    :param meta_name_y: name of y value in meta of molecule object in SDF file
    :param file: path to the SDF file
    :param max_atoms: The maximum number of atoms in the molecule
    :param properties_file: The .NPZ file containing two arrays with molecular structural properties for classification and
    regression tasks
    :return: None
    """
    accepted_atoms = ('C', 'N', 'S', 'O', 'Se', 'F', 'Cl', 'Br', 'I', 'B', 'P', 'Si')
    atoms_types = (('C', 0), ('S', 0), ('Se', 0), ('F', 0), ('Cl', 0), ('Br', 0),
                   ('I', 0), ('B', 0), ('P', 0), ('Si', 0), ('O', 0), ('O', -1), ('N', 0), ('N', 1), ('N', -1))
    mendel_info = calc_atoms_info(accepted_atoms)
    if properties_file:
        class_properties = torch.from_numpy(np.load(properties_file)['arr_0'])
        regression_properties = torch.from_numpy(np.load(properties_file)['arr_1'])
    with SDFRead(file, indexable=True) as inp:
        inp.reset_index()
        for n, molecule in tqdm(enumerate(inp), total=len(inp)):

            if len(molecule) >= max_atoms:
                raise ValueError('Found molecule with size bigger than defined')

            mol_adj, edge_attr = [], []
            for atom, neigbour, bond in molecule.bonds():
                mol_adj.append([atom - 1, neigbour - 1])
                edge_attr.append(int(bond))
            mol_adj = torch.tensor(mol_adj, dtype=torch.long)
            edge_attr = torch.tensor(edge_attr, dtype=torch.long)

            mol_atoms_x = torch.tensor(graph_to_atoms_vectors(molecule, len(molecule), mendel_info), dtype=torch.int8)
            mol_atoms_y = torch.tensor(generate_true_vector_2(molecule, atoms_types), dtype=torch.int8)
            if meta_name_y:
                if meta_name_y in molecule.meta.keys():
                    y = float(molecule.meta[meta_name_y])
                else:
                    y = 0
                mol_pyg_graph = Data(x=mol_atoms_x, edge_index=mol_adj.t().contiguous(), edge_attr=edge_attr,
                                     atoms_types=mol_atoms_y, y=y)
            elif properties_file:
                mol_pyg_graph = Data(x=mol_atoms_x, edge_index=mol_adj.t().contiguous(), edge_attr=edge_attr,
                                     atoms_types=mol_atoms_y, class_prop=class_properties[n],
                                     reg_prop=regression_properties[n])
            else:
                mol_pyg_graph = Data(x=mol_atoms_x, edge_index=mol_adj.t().contiguous(), edge_attr=edge_attr,
                                     atoms_types=mol_atoms_y)

            mol_pyg_graph = ToUndirected()(mol_pyg_graph)
            assert mol_pyg_graph.is_undirected()
            yield mol_pyg_graph


class MolDataset(InMemoryDataset, ABC):
    def __init__(self, max_atoms, molecules_file, properties_file=None, processed_path=None, meta_name_y=None):
        super().__init__(None, None, None)
        self.max_atoms = max_atoms
        self.processed_path = processed_path
        self.molecules_file = molecules_file
        self.properties_file = properties_file
        self.meta_name_y = meta_name_y
        if processed_path and osp.exists(processed_path):
            self.data, self.slices = torch.load(self.processed_path)
            print(f"Loaded preprocessed data from {processed_path}")
        else:
            self.prepare()
            print(f"Data from {molecules_file} is preprocessed")

    def prepare(self):
        processed_data = list(preprocess_molecules(self.molecules_file, self.max_atoms,
                                                   self.properties_file, self.meta_name_y))
        data, slices = self.collate(processed_data)
        if self.processed_path:
            torch.save((data, slices), self.processed_path)
        self.data = data
        self.slices = slices


class VQGAEData(LightningDataset, LightningDataModule):
    def __init__(
            self,
            path_train_predict: str,
            max_atoms: int,
            batch_size: int,
            num_workers: int = 0,
            pin_memory: bool = False,
            drop_last: bool = False,
            path_train_properties: str = None,
            path_val: str = None,
            path_val_properties: str = None,
            tmp_folder: str = None,
            meta_name_y: str = None,

    ):
        train_tmp_file = None
        val_tmp_file = None
        encode_tmp_file = None
        self.batch_size = batch_size
        if tmp_folder:
            train_tmp_file = tmp_folder + "_train.pt"
            val_tmp_file = tmp_folder + "_val.pt"
            encode_tmp_file = tmp_folder + "_encode.pt"

        val_dataset = None
        if path_val:
            train_dataset = MolDataset(max_atoms, path_train_predict, path_train_properties, train_tmp_file,
                                       meta_name_y)
            val_dataset = MolDataset(max_atoms, path_val, path_val_properties, val_tmp_file, meta_name_y)
        else:
            train_dataset = MolDataset(max_atoms, path_train_predict, path_train_properties, encode_tmp_file,
                                       meta_name_y)

        super().__init__(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

    def predict_dataloader(self) -> PYGDataLoader:
        return self.dataloader(self.train_dataset, shuffle=False, batch_size=self.batch_size)


class VQGAEVectors(LightningDataModule):
    def __init__(self, input_file, batch_size: int = 1, num_workers: int = 0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        indeces = torch.from_numpy(np.load(input_file)["arr_0"])
        self.dataset = TensorDataset(indeces)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
