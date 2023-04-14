from abc import ABC

import pytorch_lightning as pl
import torch
from adabelief_pytorch import AdaBelief
from torch.nn import Linear
from torch.nn.functional import cross_entropy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import to_dense_batch, to_dense_adj

from .layers import bonds_masking, VectorQuantizer, VQEncoder, VQDecoder, MultiTargetClassifier
from .metrics import ReconstructionRate, BondsError
from .metrics import compute_atoms_states_loss, compute_bonds_loss_v2
from .utils import convert_codebook_indices


class BaseAutoEncoder(pl.LightningModule):
    def __init__(self, batch_size: int = 512, lr: float = 2e-4, task='train'):
        super(BaseAutoEncoder, self).__init__()
        self.task = task
        self.batch_size = batch_size
        self.lr = lr
        self.encoder = None
        self.decoder = None
        self.train_bonds_error = BondsError()
        self.val_bonds_error = BondsError()
        self.train_rec_rate = ReconstructionRate()
        self.val_rec_rate = ReconstructionRate()

    def forward(self, batch):
        if self.task in ['train', 'reconstruct', 'tune_decoder']:
            return self.reconstruct(batch)
        elif self.task == 'encode':
            return self.encode(batch)
        elif self.task == 'decode':
            return self.decode(batch)

    def reconstruct(self, batch):
        raise NotImplementedError

    def encode(self, batch):
        raise NotImplementedError

    def node_embeddings(self, batch):
        raise NotImplementedError

    def decode(self, batch):
        raise NotImplementedError

    def _get_reconstruction_loss(self, batch, step, batch_idx):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        metrics = self._get_reconstruction_loss(batch, 'train', batch_idx)
        for name, value in metrics.items():
            self.log('train_' + name, value, prog_bar=True, on_step=True, on_epoch=True, batch_size=self.batch_size)
        return metrics['loss']

    def validation_step(self, batch, batch_idx):
        metrics = self._get_reconstruction_loss(batch, 'val', batch_idx)
        for name, value in metrics.items():
            self.log('val_' + name, value, on_epoch=True, batch_size=self.batch_size)

    def test_step(self, batch, batch_idx):
        metrics = self._get_reconstruction_loss(batch, 'val', batch_idx)
        for name, value in metrics.items():
            self.log('test_' + name, value, on_epoch=True, batch_size=self.batch_size)

    def configure_optimizers(self):
        if self.task != 'tune_decoder':
            parameters = self.parameters()
        else:
            parameters = self.decoder.parameters()

        optimizer = AdaBelief(parameters, lr=self.lr, eps=1e-16, betas=(0.9, 0.999),
                              weight_decouple=True, rectify=True, weight_decay=0.01,
                              print_change_log=False)

        lr_scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.8, min_lr=5e-5, verbose=True)
        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]


class VQGAE(BaseAutoEncoder, pl.LightningModule, ABC):
    def __init__(self, batch_size: int, max_atoms: int, num_conv_layers: int, vector_dim: int,
                 num_mha_layers: int, num_agg_layers: int, num_heads_encoder: int, num_heads_decoder: int,
                 dropout: float, bias: bool = True, init_values=1e-4, lr: float = 2e-4, deg=None, task='train',
                 conv_type='GCN', shuffle_graph=False, sparse_vq=True, positional_bias=False,
                 debug=False, reparam=False):
        super(VQGAE, self).__init__()
        self.save_hyperparameters()
        self.reparam = reparam
        self.debug = debug
        self.task = task
        self.lr = lr
        self.max_atoms = max_atoms
        self.batch_size = batch_size
        self.shuffle_graph = shuffle_graph
        self.sparse_vq = sparse_vq
        self.encoder = VQEncoder(max_atoms, vector_dim, batch_size, num_heads_encoder, num_conv_layers,
                                 num_agg_layers, bias, deg, dropout, init_values, aggregation=True, conv_type=conv_type)
        if task == 'tune_decoder':
            self.vq = VectorQuantizer(num_embeddings=4096, embedding_dim=vector_dim, tune=True)
        else:
            self.vq = VectorQuantizer(num_embeddings=4096, embedding_dim=vector_dim)
        self.decoder = VQDecoder(max_atoms, vector_dim, num_heads_decoder, num_mha_layers, bias,
                                 dropout, init_values, positional_bias=positional_bias)
        self.property_classifier = MultiTargetClassifier(vector_dim, [38, 29, 21, 25, 16, 13, 50, 13])
        if self.reparam:
            self.mu_lin = Linear(vector_dim, vector_dim)
            self.logvar_lin = Linear(vector_dim, vector_dim)
        self.train_bonds_error = BondsError()
        self.val_bonds_error = BondsError()
        self.train_rec_rate = ReconstructionRate()
        self.val_rec_rate = ReconstructionRate()
        self.prints = True

    def reconstruct(self, batch):
        mol_graph = batch
        atoms_vectors, _ = self.encoder(mol_graph)
        atoms_vectors, vq_loss, _ = self.vq(atoms_vectors)
        p_atoms, p_bonds = self.decoder(atoms_vectors)
        return p_atoms, p_bonds

    def encode(self, batch):
        mol_graph = batch

        t_atoms, atoms_mask = to_dense_batch(mol_graph.atoms_types.long(), mol_graph.batch,
                                             max_num_nodes=self.max_atoms, batch_size=self.batch_size)
        mol_sizes = torch.sum(atoms_mask, dim=-1)

        atoms_vectors, feature_vector, sparsity_mask = self.encoder(mol_graph, mol_sizes)
        _, _, last_dim = atoms_vectors.shape
        if self.sparse_vq:
            sparsity_mask = torch.unsqueeze(sparsity_mask.view(-1), -1)
            atoms_vectors = atoms_vectors.reshape(-1, last_dim)
            atoms_vectors = torch.masked_select(atoms_vectors, sparsity_mask).reshape(-1, last_dim)

        atoms_vectors, _, codebook_inds = self.vq(atoms_vectors)

        if self.sparse_vq:
            atoms_vectors, _ = to_dense_batch(atoms_vectors, mol_graph.batch,
                                              max_num_nodes=self.max_atoms, batch_size=self.batch_size)
            codebook_inds, _ = to_dense_batch(codebook_inds, mol_graph.batch,
                                              max_num_nodes=self.max_atoms, batch_size=self.batch_size)

        codebook_inds = codebook_inds.cpu().numpy()
        cb_inds = convert_codebook_indices(codebook_inds, mol_sizes)

        return atoms_vectors, feature_vector, cb_inds

    def decode(self, batch):
        indices = batch[0].long()
        last_ind = int(indices[0][-1])
        sizes = torch.sum(torch.where(indices != last_ind, 1, 0), -1)
        codebook_vectors = self.vq.embed_code(indices)
        p_atoms, p_bonds = self.decoder(codebook_vectors)
        atoms = torch.argmax(p_atoms, -1)
        bonds = torch.argmax(p_bonds, -3)
        return atoms, bonds, sizes

    def reparameterize(self, latents):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        mu = self.mu_lin(latents)
        logvar = self.logvar_lin(latents)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu, logvar, eps * std + mu

    def _get_reconstruction_loss(self, batch, step, batch_idx):
        mol_graph = batch
        t_atoms, atoms_mask = to_dense_batch(mol_graph.atoms_types.long(), mol_graph.batch,
                                             max_num_nodes=self.max_atoms, batch_size=self.batch_size)

        t_bonds = to_dense_adj(mol_graph.edge_index.long(), mol_graph.batch,
                               mol_graph.edge_attr.long(), max_num_nodes=self.max_atoms)

        atoms_mask = atoms_mask.long()
        adjs_mask = bonds_masking(atoms_mask)
        mol_sizes = torch.sum(atoms_mask, dim=-1)

        atoms_vectors, latents, sparsity_mask = self.encoder(mol_graph, mol_sizes)
        _, _, last_dim = atoms_vectors.shape
        if self.sparse_vq:
            sparsity_mask = torch.unsqueeze(sparsity_mask.view(-1), -1)
            atoms_vectors = atoms_vectors.reshape(-1, last_dim)
            atoms_vectors = torch.masked_select(atoms_vectors, sparsity_mask).reshape(-1, last_dim)

        atoms_vectors, vq_loss, _ = self.vq(atoms_vectors)

        if self.shuffle_graph:
            t_atoms, t_bonds, atoms_vectors = _shuffle_graph(mol_sizes, t_atoms, t_bonds, atoms_vectors)

        if self.sparse_vq:
            atoms_vectors, _ = to_dense_batch(atoms_vectors, mol_graph.batch,
                                              max_num_nodes=self.max_atoms, batch_size=self.batch_size)


        p_atoms, p_bonds = self.decoder(atoms_vectors, atoms_mask)

        atoms_loss = compute_atoms_states_loss(p_atoms, t_atoms, atoms_mask, mol_sizes)
        bonds_loss = compute_bonds_loss_v2(p_bonds, t_bonds, adjs_mask, mol_sizes)

        if self.reparam:
            mu, log_var, latents = self.reparameterize(latents)
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        property_loss = torch.zeros_like(atoms_loss, device=atoms_loss.device)
        class_properties = self.property_classifier(latents)
        true_properties = mol_graph.class_prop.view(-1, 8).long()
        for n, pred_prop in enumerate(class_properties):
            class_loss = cross_entropy(pred_prop, true_properties[:, n] + 1, reduction='none')
            property_loss += class_loss
        loss = atoms_loss + bonds_loss + torch.tensor(0.05, device=property_loss.device) * property_loss

        loss = torch.mean(loss) + torch.tensor(0.5, device=vq_loss.device) * vq_loss

        if self.reparam:
            loss = loss + torch.tensor(0.01, device=kld_loss.device) * kld_loss

        if step == 'train':
            bonds_error = self.train_bonds_error(p_bonds, t_bonds, adjs_mask)
            rec_rate = self.train_rec_rate((p_atoms, p_bonds,),
                                           (t_atoms, t_bonds,),
                                           (atoms_mask, adjs_mask,))
        elif step == 'val':
            bonds_error = self.val_bonds_error(p_bonds, t_bonds, adjs_mask)
            rec_rate = self.val_rec_rate((p_atoms, p_bonds,),
                                         (t_atoms, t_bonds,),
                                         (atoms_mask, adjs_mask,))

        metrics = {'loss': loss, 'atoms_loss': torch.mean(atoms_loss),
                   'bonds_loss': torch.mean(bonds_loss), 'vq_loss': torch.mean(vq_loss),
                   'property_loss': torch.mean(property_loss),
                   'rec_rate': rec_rate, 'bonds_error': bonds_error}

        if self.reparam:
            metrics['kld_loss'] = kld_loss

        return metrics


def _create_random_permutations(sizes, batch, max_atoms, device):
    permuted_indices = torch.zeros((batch, max_atoms), dtype=torch.long, device=device)
    for i in range(batch):
        random_real = torch.randperm(sizes[i])
        mask_indices = torch.arange(sizes[i], max_atoms)
        permuted_indices[i] = torch.cat([random_real, mask_indices])
    return torch.unsqueeze(permuted_indices, -1)


def _permute_adj_matrix(indices, adjs, max_atoms):
    adj_perm = indices.repeat(1, 1, max_atoms)
    shuffled_adjs = torch.gather(adjs, -2, adj_perm)
    shuffled_adjs = torch.gather(shuffled_adjs, -1, adj_perm.permute(0, 2, 1))
    return shuffled_adjs


def _shuffle_graph(sizes, atoms, adjs, atoms_vectors):
    batch, max_atoms, dim = atoms_vectors.shape
    permuted_indices = _create_random_permutations(sizes, batch, max_atoms, atoms.device)
    shuffled_atoms = torch.gather(atoms, 1, permuted_indices)
    shuffled_adjs = _permute_adj_matrix(permuted_indices, adjs, max_atoms)

    vectors_perm = permuted_indices.repeat(1, 1, dim)
    shuffled_vectors = torch.gather(atoms_vectors, 1, vectors_perm)

    return shuffled_atoms, shuffled_adjs, shuffled_vectors
