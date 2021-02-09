import typing
from torch_geometric.nn import GCNConv

from torch import nn


class GCN(nn.Module):
    def __init__(self, num_features, num_units, num_classes, dropout=0.5):
        super().__init__()
        self.layer1 = GCNConv(num_features, num_units, cached=True)
        self.layer2 = GCNConv(num_units, num_classes, cached=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def __call__(self, *args, **kwargs) -> typing.Any:
        return super().__call__(*args, **kwargs)

    def forward(self, features, indices, emb=False):
        out1 = self.layer1(features, indices)
        out2 = self.relu(out1)
        out2 = self.dropout(out2)
        out2 = self.layer2(out2, indices)
        if emb:
            return out1, out2
        return out2
