import torch
from torch import nn


class MoGCL(nn.Module):
    def __init__(self, base_encoder, dim=64, k=65536, m=0.999, T=0.07, mlp=True):
        """
        dim: feature dimension (default: 64)
        K: queue size; number of negative keys (default: 65536)
        m: MoGCL momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoGCL, self).__init__()
        self.k = k
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:
            pass

