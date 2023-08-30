import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import (
    ModuleList,
    Linear,
    Module,
    Parameter,
    GRU,
    Dropout,
    LayerNorm,
    Embedding,
    GELU
)
from torch.nn import functional
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import to_dense_batch


def bonds_masking(atoms_mask):
    atoms_mask_ext = torch.unsqueeze(atoms_mask, -1)
    adj_mask = atoms_mask_ext * atoms_mask_ext.permute(0, 2, 1)
    adj_mask -= torch.diag_embed(atoms_mask)
    return adj_mask


def graph_masking(atoms: torch.Tensor):
    atoms_mask = torch.where(atoms[:, :, 0] > -1, 1, 0)
    atoms_mask_ext = torch.unsqueeze(atoms_mask, -1)
    adj_mask = atoms_mask_ext * atoms_mask_ext.permute(0, 2, 1)
    adj_mask -= torch.diag_embed(atoms_mask)
    return atoms_mask, adj_mask


class MultiTargetClassifier(Module):
    def __init__(self, input_shape: int, num_classes: list):
        """

        :param input_shape: shape of input vector
        :param num_classes: number of classes for each target
        """
        super().__init__()
        layers = []
        for n in num_classes:
            layers.append(Linear(input_shape, n))
        self.layers = ModuleList(layers)

    def forward(self, inputs):
        predictions = []
        for layer in self.layers:
            predictions.append(layer(inputs))
        return predictions


class Defactorization(Module):
    def __init__(self, vector_length, bias=True):
        super(Defactorization, self).__init__()
        self.weights = nn.Parameter(torch.empty((vector_length,)))
        with torch.no_grad():
            nn.init.normal_(self.weights)
        self.bias = bias
        if bias:
            self.bias_weights = nn.Parameter(torch.empty((vector_length,)))
            with torch.no_grad():
                nn.init.zeros_(self.bias_weights)
        else:
            self.register_parameter("bias_weights", None)

    def forward(self, inputs):
        updated_vec = torch.mul(inputs, self.weights)
        if self.bias:
            updated_vec += self.bias_weights
        adj = torch.matmul(updated_vec, inputs.permute(0, 2, 1))
        return adj


class MultiEdgeDefactorization(Module):
    def __init__(self, max_atoms, vector_dim, bias=False):
        super(MultiEdgeDefactorization, self).__init__()
        self.bonds_linear = ModuleList(
            [Linear(vector_dim, max_atoms) for _ in range(4)]
        )
        self.bonds_weights = Parameter(torch.empty((4, max_atoms)))
        with torch.no_grad():
            nn.init.normal_(self.bonds_weights)
        self.bias = bias
        if bias:
            self.bonds_bias = Parameter(torch.zeros(4, max_atoms))
        else:
            self.register_parameter("bonds_bias", None)

    def forward(self, inputs):
        inputs_transformed = []
        for layer in self.bonds_linear:
            inputs_transformed.append(layer(inputs))
        inputs = torch.stack(inputs_transformed, -2)
        bonds = torch.mul(inputs, self.bonds_weights)
        if self.bias:
            bonds += self.bonds_bias
        bonds = torch.matmul(bonds.permute(0, 2, 1, 3), inputs.permute(0, 2, 3, 1))
        return bonds


class AttentionTalkingHead(Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add Talking Heads Attention (https://arxiv.org/pdf/2003.02436v1.pdf)
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
    ):
        super().__init__()

        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_l = Linear(num_heads, num_heads)
        self.proj_w = Linear(num_heads, num_heads)
        self.proj_drop = Dropout(proj_drop)

    def forward(self, x):
        batch, max_atoms, dim = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch, max_atoms, 3, self.num_heads, dim // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = q @ k.transpose(-2, -1)

        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attn = attn.softmax(dim=-1)

        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch, max_atoms, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FFN(Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerScaleBlock(Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add layerScale
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            act_layer=nn.GELU,
            init_values=1e-4,
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = AttentionTalkingHead(
            dim, num_heads, qkv_bias, qk_scale, attn_drop, drop
        )
        self.norm2 = LayerNorm(dim)
        self.ffn = FFN(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.gamma_1 = Parameter(init_values * torch.ones((dim,)), requires_grad=True)
        self.gamma_2 = Parameter(init_values * torch.ones((dim,)), requires_grad=True)

    def forward(self, x):
        x = x + self.gamma_1 * self.attn(self.norm1(x))
        x = x + self.gamma_2 * self.ffn(self.norm2(x))
        return x


class GraphEmbedding(Module):
    def __init__(
            self,
            max_atoms: int,
            vector_dim: int,
            batch: int,
            bias: bool = True,
            num_conv_layers: int = 5,
            *args,
            **kwargs
    ):
        super(GraphEmbedding, self).__init__(*args, **kwargs)
        self.max_atoms = max_atoms
        self.batch = batch
        self.expansion = Linear(11, vector_dim, bias)
        self.gcn_convs = ModuleList(
            [
                GCNConv(vector_dim, vector_dim, improved=True)
                for _ in range(num_conv_layers)
            ]
        )

    def forward(self, atoms, connections, batch):
        atoms = torch.log(atoms + 1)
        atoms = self.expansion(atoms)

        for gcn_conv in self.gcn_convs:
            atoms = gcn_conv(atoms, connections)
            atoms = functional.relu(atoms)

        atoms, mask = to_dense_batch(
            atoms, batch, max_num_nodes=self.max_atoms, batch_size=self.batch
        )
        return atoms, mask


class GraphReconstruction(Module):
    def __init__(
            self, max_atoms: int, vector_dim: int, bias: bool = True, *args, **kwargs
    ):
        super(GraphReconstruction, self).__init__(*args, **kwargs)
        self.predict_atoms = Linear(vector_dim // 2, 15, bias)
        self.deembedding = Linear(vector_dim, vector_dim // 2, True)
        self.factorize = Linear(vector_dim, vector_dim // 8, True)
        self.predict_bonds = MultiEdgeDefactorization(max_atoms, vector_dim // 8, bias)

    def forward(self, inputs):
        atoms_vectors = self.deembedding(inputs)
        atoms_vectors = functional.gelu(atoms_vectors)
        p_atoms = self.predict_atoms(atoms_vectors)
        bonds_vectors = self.factorize(inputs)
        p_bonds = self.predict_bonds(bonds_vectors)

        return p_atoms, p_bonds


class VQGraphAggregation(Module):
    def __init__(
            self,
            vector_dim: int,
            num_heads: int = 16,
            num_agg_layers: int = 3,
            dropout: float = 0.1,
            init_values=1e-2,
            *args,
            **kwargs
    ):
        super(VQGraphAggregation, self).__init__(*args, **kwargs)
        self.self_attention = ModuleList(
            [
                LayerScaleBlock(
                    dim=vector_dim,
                    num_heads=num_heads,
                    drop=dropout,
                    init_values=init_values,
                )
                for _ in range(num_agg_layers)
            ]
        )

    def forward(self, atoms_vectors, graph_sizes):
        batch, _, _ = atoms_vectors.shape
        summed_vector = torch.sum(atoms_vectors, 1, keepdim=True)
        atoms_vectors = torch.cat((summed_vector, atoms_vectors), 1)
        for layer in self.self_attention:
            atoms_vectors = layer(atoms_vectors)
        feature_vector = atoms_vectors[:, 0, :]
        new_atoms_vectors = atoms_vectors[:, 1:, :]
        return new_atoms_vectors, feature_vector


class VQEncoder(Module):
    def __init__(
            self,
            max_atoms: int,
            vector_dim: int,
            batch_size: int,
            num_heads: int = 16,
            num_conv_layers: int = 5,
            num_agg_layers: int = 2,
            bias: bool = True,
            dropout=0.1,
            init_values=1e-4,
            class_aggregation=False,
            *args,
            **kwargs
    ):
        super(VQEncoder, self).__init__(*args, **kwargs)
        self.graph_embedding = GraphEmbedding(
            max_atoms=max_atoms,
            vector_dim=vector_dim,
            batch=batch_size,
            bias=bias,
            num_conv_layers=num_conv_layers
        )
        self.class_aggregation = class_aggregation
        if self.class_aggregation:
            self.graph_aggregation = VQGraphAggregation(
                vector_dim=vector_dim,
                num_heads=num_heads,
                num_agg_layers=num_agg_layers,
                dropout=dropout,
                init_values=init_values,
            )

    def forward(self, graph, mol_sizes):
        atoms, connections = graph.x.float(), graph.edge_index.long()
        atoms_embeddings, sparsity_mask = self.graph_embedding(
            atoms, connections, graph.batch
        )
        if self.class_aggregation:
            atoms_vectors, feature_vector = self.graph_aggregation(atoms_embeddings, mol_sizes)
        else:
            atoms_vectors = atoms_embeddings
            feature_vector = torch.sum(atoms_embeddings, axis=-2)
            feature_vector /= torch.unsqueeze(mol_sizes, -1)
        return atoms_vectors, feature_vector, sparsity_mask


class VQDecoder(Module):
    def __init__(
            self,
            max_atoms: int,
            vector_dim: int,
            num_heads: int,
            num_mha_layers: int,
            bias: bool = True,
            dropout: float = 0.1,
            init_values: float = 1e-4,
            positional_bias=False,
            *args,
            **kwargs
    ):
        super(VQDecoder, self).__init__(*args, **kwargs)
        self.positional_bias = positional_bias
        if self.positional_bias:
            self.graph_bias = nn.Parameter(
                torch.zeros((1, max_atoms, vector_dim,)),
                requires_grad=True,
            )
        else:
            self.rnn = GRU(vector_dim, vector_dim)
            self.gamma_rnn = Parameter(
                init_values * torch.ones((vector_dim,)),
                requires_grad=True,
            )
            self.self_attention = ModuleList([
                LayerScaleBlock(vector_dim, num_heads, drop=dropout, init_values=init_values, )
                for _ in range(num_mha_layers)
            ])
            self.graph_reconstruction = GraphReconstruction(max_atoms, vector_dim, bias)
            self.rnn_dropout = Dropout(dropout)

    def forward(self, atoms_vectors: torch.Tensor):
        if self.positional_bias:
            repeated_graph_bias = self.graph_bias.expand(atoms_vectors.shape[0], -1, -1)
            atoms_vectors += repeated_graph_bias
        else:
            ordered_vectors, _ = self.rnn(atoms_vectors.permute(1, 0, 2))
            ordered_vectors = ordered_vectors.permute(1, 0, 2)
            atoms_vectors = atoms_vectors + self.gamma_rnn * self.rnn_dropout(ordered_vectors)
        for layer in self.self_attention:
            atoms_vectors = layer(atoms_vectors)
        p_atoms, p_bonds = self.graph_reconstruction(atoms_vectors)
        return p_atoms, p_bonds


class VectorQuantizer(Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            flatten: bool = False,
            tune: bool = False,
            ema: bool = True,
    ):
        super().__init__()
        self.n_embed = num_embeddings
        self.dim = embedding_dim
        self.decay = 0.99
        self.eps = 1e-5
        self.flatten = flatten
        self.tune = tune
        self.ema = ema
        self.build()

    def build(self):
        # workaround to register buffer variables
        codebook = torch.empty(self.dim, self.n_embed, dtype=torch.float32)
        torch.nn.init.uniform_(codebook, a=-1.0 / self.n_embed, b=1.0 / self.n_embed)
        self.register_buffer("codebook", codebook)
        self.register_buffer(
            "cluster_size", torch.zeros(self.n_embed, dtype=torch.float32)
        )
        self.register_buffer("embed_avg", self.codebook.clone())

    @torch.cuda.amp.autocast(enabled=False)
    def forward(
            self, latents: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, float, torch.LongTensor]:

        if self.flatten:
            latents = latents.reshape(-1, self.dim)

        dist = (
                latents.pow(2).sum(1, keepdim=True)
                - 2 * latents @ self.codebook
                + self.codebook.pow(2).sum(0, keepdim=True)
        )

        _, codebook_ind = (-dist).max(1)
        codebook_onehot = functional.one_hot(codebook_ind, self.n_embed).type(latents.dtype)
        codebook_ind = codebook_ind.view(*latents.shape[:-1])
        quantize = self.embed_code(codebook_ind)
        quantize = latents + (quantize - latents).detach()
        if self.ema:
            vq_loss = (quantize.detach() - latents).pow(2).mean()

            if self.training and not self.tune:
                # EMA algorithm
                codebook_onehot_sum = codebook_onehot.sum(0)
                codebook_sum = latents.transpose(0, 1) @ codebook_onehot
                self.cluster_size.data.mul_(self.decay).add_(
                    codebook_onehot_sum, alpha=1 - self.decay
                )
                self.embed_avg.data.mul_(self.decay).add_(
                    codebook_sum, alpha=1 - self.decay
                )

                n = self.cluster_size.sum()
                cluster_size = (
                        (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
                )
                embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
                self.codebook.data.copy_(embed_normalized)
        else:
            commitment_loss = functional.mse_loss(quantize.detach(), latents)
            embedding_loss = functional.mse_loss(quantize, latents.detach())
            vq_loss = commitment_loss * self.beta + embedding_loss

        return quantize.contiguous(), vq_loss, codebook_ind

    def embed_code(self, embed_id: torch.LongTensor):
        # embed_id - (batch, max number of atoms)
        return functional.embedding(embed_id, self.codebook.transpose(0, 1))


class PositionalEncoding(Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 51):
        super().__init__()
        self.dropout = Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.permute(1, 0, 2)
        x = x + self.pe[: x.size(0)]
        return self.dropout(x).permute(1, 0, 2)


class ONN(Module):
    def __init__(
            self,
            mol_size: int,
            num_embeddings: int,
            embedding_dim: int = 128,
            num_heads=8,
            num_mha=2,
            init_values=0.001,
            dropout: float = 0.2,
    ):
        """
        """
        super().__init__()
        self.emb = Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.positional_encoding = Parameter(torch.empty((1, mol_size, embedding_dim)), requires_grad=True)
        with torch.no_grad():
            torch.nn.init.normal_(self.positional_encoding)
        self.attentions = ModuleList([
            LayerScaleBlock(
                embedding_dim,
                num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                act_layer=GELU,
                init_values=init_values,
                drop=dropout
            ) for _ in range(num_mha)
        ])
        self.scorer = Linear(embedding_dim, mol_size)

    def forward(self, inputs, mask):
        x = self.emb(inputs) + self.positional_encoding
        x = x * mask
        for layer in self.attentions:
            x = layer(x)
        scores = self.scorer(x)
        return scores
