import torch
from torch import nn


class Contrast(nn.Module):
    def __init__(self, hidden_dim):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self):
        pass
