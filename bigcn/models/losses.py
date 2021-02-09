from torch import nn


def bdot(a, b):
    """
    Perform the dot-product for each vector in a batch.
    """
    return a.unsqueeze(1).bmm(b.unsqueeze(2)).view(-1)


class LaplacianLoss(nn.Module):
    """
    Normal Laplacian loss (for baseline models).
    """

    def __init__(self):
        super().__init__()

    def forward(self, features, indices):
        x_squared = bdot(features, features)
        x1 = x_squared[indices[0]]
        x2 = x_squared[indices[1]]
        x3 = bdot(features[indices[0]], features[indices[1]])
        return (x1 + x2 - 2 * x3).mean()
