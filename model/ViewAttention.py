import torch
from .Attention import Attention


class ViewAttention(Attention):
    def __init__(self, hidden_dim, attn_dim, attn_drop):
        super(ViewAttention, self).__init__(hidden_dim, attn_dim, attn_drop)

    def forward(self, inputs):
        pass
