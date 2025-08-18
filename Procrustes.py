import torch
from torch.nn import Module

class ProcrustesDistance(Module):
    def __init__(self, centered=True):
        super(ProcrustesDistance, self).__init__()
        self.centered = centered

    def forward(self, X, Y):
        """
        Compute the Procrustes distance between two matrices X and Y.
        """
        X = torch.as_tensor(X, dtype=torch.float64)
        Y = torch.as_tensor(Y, dtype=torch.float64)
        if X.device != Y.device:
            Y = Y.to(X.device)
        X_mean = X.mean(dim=0, keepdim=True)
        Y_mean = Y.mean(dim=0, keepdim=True)
        if self.centered:
            X = X - X_mean
            Y = Y - Y_mean
        norm_X = torch.linalg.norm(X)**2
        norm_Y = torch.linalg.norm(Y)**2
        M = X.T @ Y
        s = torch.linalg.svdvals(M).sum()
        d = norm_X + norm_Y - 2.0 * s
        return torch.sqrt(d)