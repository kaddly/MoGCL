import torch
from torch import nn
from .NeighborEncoder import NeighborEncoder
from .ViewAttention import ViewAttention


class FD_GCL(nn.Module):
    def __init__(self):
        super(FD_GCL, self).__init__()
        self.neighbor_encoder = NeighborEncoder()
