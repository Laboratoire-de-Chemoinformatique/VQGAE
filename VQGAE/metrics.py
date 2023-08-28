import torch
from torch.nn.functional import (
    cross_entropy,
    one_hot,
    mse_loss,
    binary_cross_entropy_with_logits,
)
from torchmetrics import Metric


def compute_atoms_loss(pred, true, mask, length):
    """
    Compute the cross entropy loss of the predicted states and true states.

    :param pred: the predicted states
    :param true: the ground truth, which is a tensor of shape (batch_size, max_length, num_atoms)
    :param mask: a tensor of shape (batch_size, max_length)
    :param length: the length of the sequence
    :return: The loss for each atom.
    """
    true *= mask
    pred = pred.permute(0, 2, 1)
    result = cross_entropy(pred, true, reduction="none")
    masked_result = result * mask
    loss = torch.sum(masked_result, dim=-1)
    loss /= length
    return loss


def compute_adj_loss(pred_adj, true_adj, adj_mask, length):
    """
    Compute the loss of the adjacency matrix.
    The loss is computed by element-wise product between the predicted adjacency matrix and the true
    adjacency matrix, followed by a sum over the entire matrix, and a normalization w.r.t. the number of edges (twice
    the number of edges because the adjacency matrix is symmetric).

    :param pred_adj: The predicted adjacency matrix
    :param true_adj: The true adjacency matrix
    :param adj_mask: (batch_size, max_length, max_length)
    :param length: the length of the sequence
    :return: The loss is the average loss over all the nodes in the graph.
    """
    result = binary_cross_entropy_with_logits(pred_adj, true_adj.float(), reduction="none")
    masked_result = result * adj_mask
    loss = torch.sum(masked_result, dim=(1, 2,),)
    loss /= length * 2
    return loss


def compute_bonds_loss(pred_bonds, true_bonds, adj_mask, length):
    true_bonds = one_hot(true_bonds, 4).float().permute(0, 3, 1, 2)
    result = binary_cross_entropy_with_logits(pred_bonds, true_bonds, reduction="none")
    adj_mask = torch.unsqueeze(adj_mask, 1)
    masked_result = result * adj_mask
    loss = torch.sum(masked_result, dim=(1, 2, 3))
    loss /= length * 2
    return loss


def compute_degree_loss(pred_adj, true_adj, atoms_mask, length):
    pred_degree = torch.sum(torch.sigmoid(pred_adj), dim=-1)
    one_hot_bonds = one_hot(true_adj, num_classes=4).permute(0, 3, 1, 2)
    true_degree = torch.sum(one_hot_bonds, dim=-1)
    result = mse_loss(pred_degree, true_degree.float(), reduction="none")
    atoms_mask = torch.unsqueeze(atoms_mask, -2)
    masked_result = result * atoms_mask
    loss = torch.mean(masked_result, dim=(1, 2), keepdim=False)
    # loss /= (length * 2)
    return loss


def compute_atoms_error(pred, target, mask):
    pred_chosen = torch.argmax(pred, dim=-1) * mask
    error = torch.sum(pred_chosen != target, dim=-1)
    return error.float()


def compute_adj_error(pred, target, mask):
    pred_chosen = torch.where(torch.sigmoid(pred) > 0.5, 1, 0) * mask
    error = torch.sum(pred_chosen != target, dim=(-1, -2)) / torch.sum(
        mask > 0, dim=(-1, -2)
    )
    return error.float()


def compute_bonds_error(pred, target, mask):
    pred_chosen = torch.argmax(pred, dim=1) * mask
    error = torch.sum(pred_chosen != target, dim=(-1, -2)).float()
    return error


def compute_permutaion_loss(self, perm, mask, eps=10e-8):
    def entropy(p, axis, normalize=True, eps=10e-12):
        if normalize:
            p = p / (p.sum(axis=axis, keepdim=True) + eps)
        e = -torch.sum(p * torch.clamp_min(torch.log(p), -100), axis=axis)
        return e

    perm = perm + eps
    entropy_col = entropy(perm, axis=1, normalize=True) * mask
    entropy_row = entropy(perm, axis=2, normalize=True) * mask
    loss = entropy_col.mean() + entropy_row.mean()
    return loss


class ReconstructionRate(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state(
            "rec_rate", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum"
        )
        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum"
        )

    def update(self, preds, targets, masks):
        p_atoms, p_bonds = preds
        t_atoms, t_bonds = targets
        atoms_mask, adjs_mask = masks
        atoms_error = compute_atoms_error(p_atoms, t_atoms, atoms_mask)
        adj_error = compute_bonds_error(p_bonds, t_bonds, adjs_mask)
        error = atoms_error + adj_error
        rec_rate = torch.mean(torch.where(error != 0, 0.0, 1.0))
        self.rec_rate += rec_rate
        self.total += 1.0

    def compute(self):
        return self.rec_rate / self.total


class BondsError(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state(
            "error", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum"
        )
        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum"
        )

    def update(self, pred, target, mask):
        self.error += torch.mean(compute_bonds_error(pred, target, mask))
        self.total += 1.0

    def compute(self):
        return self.error / self.total
