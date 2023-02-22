import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_dim, attn_drop):
        super(Attention, self).__init__()
        self.fc = nn.Parameter(torch.empty(hidden_dim, attn_dim))
        nn.init.xavier_normal_(self.fc, gain=1.414)
        self.trans_weights = nn.Parameter(torch.empty(attn_dim, 1))
        nn.init.xavier_normal_(self.trains_w, gain=1.414)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, inputs):
        raise NotImplementedError
