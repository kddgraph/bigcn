import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric import datasets


def preprocess_edges(edges):
    """
    Preprocess edges to make sure the following:
    1) No self-loops.
    2) Each pair (a, b) and (b, a) exists exactly once.
    """
    m = defaultdict(lambda: set())
    for src, dst in edges.t():
        src = src.item()
        dst = dst.item()
        if src != dst:
            m[src].add(dst)
            m[dst].add(src)

    edges = []
    for src in sorted(m):
        for dst in sorted(m[src]):
            edges.append((src, dst))
    return np.array(edges, dtype=np.int64).transpose()


def split_nodes(labels, trn_ratio, seed=0):
    """
    Split nodes into training, validation, and test.
    """
    state = np.random.RandomState(seed)
    all_nodes = np.arange(labels.shape[0])

    pos_nodes = all_nodes[labels == 1]
    neg_nodes = all_nodes[labels == 0]

    n_pos_nodes = pos_nodes.shape[0]
    n_neg_nodes = neg_nodes.shape[0]
    n_trn_nodes = int(n_pos_nodes * trn_ratio)
    n_val_pos_nodes = (n_pos_nodes - n_trn_nodes) // 2
    n_val_neg_nodes = n_neg_nodes // 2

    trn_nodes = state.choice(pos_nodes, size=n_trn_nodes, replace=False)

    val_pos_candidates = set(pos_nodes).difference(set(trn_nodes))
    val_pos_nodes = state.choice(list(val_pos_candidates), size=n_val_pos_nodes, replace=False)
    val_neg_nodes = state.choice(neg_nodes, size=n_val_neg_nodes, replace=False)
    val_nodes = np.concatenate([val_pos_nodes, val_neg_nodes])

    test_pos_nodes = np.array(list(val_pos_candidates.difference(set(val_pos_nodes))))
    test_neg_nodes = np.array(list(set(neg_nodes).difference(set(val_neg_nodes))))
    test_nodes = np.concatenate([test_pos_nodes, test_neg_nodes])

    return trn_nodes, val_nodes, test_nodes


def load_data(dataset, trn_ratio, verbose=False, seed=0):
    """
    Read a dataset based on its name.
    """
    root = 'data'
    root_cached = os.path.join(root, 'cached', dataset)
    if not os.path.exists(root_cached):
        if dataset == 'cora':
            data = datasets.Planetoid(root, 'Cora')
        elif dataset == 'citeseer':
            data = datasets.Planetoid(root, 'CiteSeer')
        elif dataset == 'pubmed':
            data = datasets.Planetoid(root, 'PubMed')
        elif dataset == 'cora-ml':
            data = datasets.CitationFull(root, 'Cora_ML')
        elif dataset == 'dblp':
            data = datasets.CitationFull(root, 'DBLP')
        elif dataset == 'amazon':
            data = datasets.Amazon(os.path.join(root, 'Amazon'), 'Photo')
        else:
            raise ValueError(dataset)

        node_x = data.data.x
        node_x[node_x.sum(dim=1) == 0] = 1
        node_x = node_x / node_x.sum(dim=1, keepdim=True)
        node_y = data.data.y
        edges = preprocess_edges(data.data.edge_index)

        os.makedirs(root_cached, exist_ok=True)
        np.save(os.path.join(root_cached, 'x'), node_x)
        np.save(os.path.join(root_cached, 'y'), node_y)
        np.save(os.path.join(root_cached, 'edges'), edges)

    edges = np.load(os.path.join(root_cached, 'edges.npy'))
    node_x = np.load(os.path.join(root_cached, 'x.npy'))
    node_y = np.load(os.path.join(root_cached, 'y.npy'))

    indices = np.arange(node_x.shape[0])
    trn_nodes, test_nodes = train_test_split(
        indices, test_size=0.1000, random_state=seed, stratify=node_y)
    trn_nodes, val_nodes = train_test_split(
        trn_nodes, test_size=0.1111, random_state=seed, stratify=node_y[trn_nodes])
    trn_nodes, _ = train_test_split(
        trn_nodes, train_size=trn_ratio / 0.8, random_state=seed, stratify=node_y[trn_nodes])

    edges = torch.from_numpy(edges)
    node_x = torch.from_numpy(node_x)
    node_y = torch.from_numpy(node_y)
    trn_nodes = torch.from_numpy(trn_nodes)
    val_nodes = torch.from_numpy(val_nodes)
    test_nodes = torch.from_numpy(test_nodes)

    if verbose:
        print('Number of nodes:', node_x.size(0))
        print('Number of features:', node_x.size(1))
        print('Number of edges:', edges.size(1) // 2)
        print('Number of classes:', node_y.max().item() + 1)
    return edges, node_x, node_y, trn_nodes, val_nodes, test_nodes
