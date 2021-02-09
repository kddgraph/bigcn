import os

import torch
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from torch import nn


class OneHot(nn.Module):
    def __init__(self, num_nodes, obs_nodes, obs_features):
        super().__init__()
        num_estimates = num_nodes - len(obs_nodes)
        indices = obs_features.nonzero().t()

        def build_indices():
            num_features = obs_features.size(1)
            x_mask = torch.zeros(num_nodes, dtype=torch.bool)
            x_mask[obs_nodes] = 1
            indices1 = indices.clone()
            indices1[0] = obs_nodes[indices1[0]]
            indices21 = (~x_mask).nonzero().view(-1)
            indices22 = torch.arange(num_estimates) + num_features
            indices2 = torch.stack([indices21, indices22], dim=0)
            return torch.cat((indices1, indices2), dim=1)

        def build_values():
            values1 = obs_features[indices[0], indices[1]]
            values2 = torch.ones(num_estimates)
            return torch.cat((values1, values2))

        self.indices = nn.Parameter(build_indices(), requires_grad=False)
        self.values = nn.Parameter(build_values(), requires_grad=False)
        self.features = None
        self.num_features = obs_features.size(1) + num_estimates

    def forward(self):
        device = self.indices.device
        if self.features is None:
            self.features = torch.sparse_coo_tensor(self.indices, self.values, device=device)
        elif self.features.device != device:
            self.features = self.features.to(device)
        return self.features


class Param(nn.Module):
    def __init__(self, num_nodes, x_nodes, x_features):
        super().__init__()
        self.features = nn.Parameter(torch.zeros(num_nodes, x_features.size(1)), requires_grad=True)
        self.seeds = nn.Parameter(x_nodes, requires_grad=False)
        self.seed_features = nn.Parameter(x_features, requires_grad=False)
        self.num_features = x_features.size(1)

    def forward(self):
        return self.features.index_copy(0, self.seeds, self.seed_features)


class Embedding(nn.Module):
    path = '../emb'

    def __init__(self, model, graph, seed=0):
        super().__init__()
        emb = np.load(os.path.join(self.path, model, graph, str(seed), 'emb.npy'))
        self.emb = nn.Parameter(torch.from_numpy(emb), requires_grad=False)
        self.num_features = emb.shape[1]

    def forward(self):
        return self.emb


class Generator(nn.Module):
    def __init__(self, graph, edges, num_nodes, obs_nodes, obs_features, mode='param'):
        super().__init__()
        if mode == 'onehot':
            self.model = OneHot(num_nodes, obs_nodes, obs_features)
        elif mode == 'svd':
            self.model = SVD(num_nodes, edges, rank=16)
        elif mode in ['param', 'laplacian']:
            self.model = Param(num_nodes, obs_nodes, obs_features)
        elif mode in ['node2vec']:
            self.model = Embedding(mode, graph)
        else:
            raise ValueError()

    def forward(self):
        return self.model()

    def num_features(self):
        return self.model.num_features


class SVD(nn.Module):
    def __init__(self, num_nodes, graph, rank):
        super().__init__()
        if isinstance(graph, torch.Tensor):
            graph = graph.cpu().numpy()
        adj = csc_matrix((np.ones(graph.shape[1]), graph), (num_nodes, num_nodes))
        # noinspection PyTypeChecker
        u, s, _ = svds(adj, rank)
        features = np.matmul(u, np.diag(s)).astype(np.float32)
        self.features = nn.Parameter(torch.from_numpy(features), requires_grad=False)
        self.num_features = features.shape[1]

    def forward(self):
        return self.features
