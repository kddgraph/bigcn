import torch
import typing
import numpy as np
from torch import nn


def to_parameter(tensor):
    if tensor is None:
        return None
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    return nn.Parameter(tensor, requires_grad=False)


class BPModel(nn.Module):
    """
    Model that runs loopy belief propagation (LBP) over an undirected graph.

    It assumes the following:
    a) The edges [(src, dst)] are sorted well by the source and destination edges.
    b) Both directions (src, dst) and (dst, src) of each edge must exist together.
    """

    def __init__(self, edges, num_nodes, num_classes, obs_nodes=None, obs_labels=None,
                 epsilon=2., num_iters=64):
        super().__init__()

        self.num_iters = num_iters
        self.num_nodes = num_nodes
        self.num_edges = edges.size(1)
        self.num_classes = num_classes
        self.softmax = nn.Softmax(dim=1)
        self.threshold = 1e-8

        self.potential = to_parameter(torch.exp(torch.eye(num_classes) * epsilon))
        self.obs_nodes = to_parameter(obs_nodes)
        self.obs_labels = to_parameter(obs_labels)
        self.src_nodes = to_parameter(edges[0, :])
        self.dst_nodes = to_parameter(edges[1, :])
        self.rev_edges = to_parameter(self.to_rev_edges())

    def __call__(self, *args, **kwargs) -> typing.Any:
        return super().__call__(*args, **kwargs)

    def to_rev_edges(self):
        degrees = torch.zeros(self.num_nodes, dtype=torch.int64)
        degrees.index_add_(0, self.src_nodes, torch.ones(self.num_edges, dtype=torch.int64))
        indices = torch.cat([torch.zeros(1, dtype=torch.int64), degrees.cumsum(dim=0)[:-1]])
        counts = torch.zeros(self.num_nodes, dtype=torch.int64)
        rev_edges = torch.zeros(self.num_edges, dtype=torch.int64)
        for edge_idx in range(self.num_edges):
            src = self.dst_nodes[edge_idx]
            rev_edges[indices[src] + counts[src]] = edge_idx
            edge_idx += 1
            counts[src] += 1
        return rev_edges

    def update_messages(self, messages, beliefs):
        new_beliefs = beliefs[self.src_nodes, :]
        rev_messages = messages[self.rev_edges, :]
        new_msgs = torch.mm(new_beliefs / rev_messages, self.potential)
        return new_msgs / new_msgs.sum(dim=1, keepdim=True)

    def compute_beliefs(self, messages, priors):
        beliefs = priors.log()
        beliefs = beliefs.index_add(0, self.dst_nodes, messages.log())
        return self.softmax(beliefs)

    def device(self):
        return self.potential.device

    def initialize_priors(self):
        assert self.obs_nodes is not None and self.obs_labels is not None
        device = self.device()
        num_obs_nodes = len(self.obs_nodes)
        onehot = torch.zeros(num_obs_nodes, self.num_classes, device=device)
        onehot[torch.arange(num_obs_nodes), self.obs_labels] = 1
        priors = torch.full((self.num_nodes, self.num_classes), 1 / self.num_classes, device=device)
        return priors.index_copy(0, self.obs_nodes, onehot)

    def forward(self, priors=None):
        device = self.device()
        if priors is None:
            priors = self.initialize_priors().to(device)
        beliefs = priors
        messages = torch.full([self.num_edges, self.num_classes], 1 / self.num_classes, device=device)
        for _ in range(self.num_iters):
            old_beliefs = beliefs
            messages = self.update_messages(messages, beliefs)
            beliefs = self.compute_beliefs(messages, priors)
            diff = (beliefs - old_beliefs).abs().max()
            if diff < self.threshold:
                break
        return beliefs
