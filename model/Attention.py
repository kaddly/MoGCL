import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_dim, attn_drop):
        super(Attention, self).__init__()
        self.fc = nn.Parameter(torch.empty(hidden_dim, attn_dim), requires_grad=True)
        nn.init.xavier_normal_(self.fc, gain=1.414)
        self.trans_weights = nn.Parameter(torch.empty(attn_dim, 1), requires_grad=True)
        nn.init.xavier_normal_(self.trans_weights, gain=1.414)

        self.tanh = nn.Tanh()
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, inputs):
        raise NotImplementedError


class NeighborEncoder(Attention):
    def __init__(self, hidden_dim, attn_dim, attn_drop):
        super(NeighborEncoder, self).__init__(hidden_dim, attn_dim, attn_drop)

    def forward(self, inputs):
        # inputs.shape:(batch_size, num_view, num_neighbor, hidden_dim)
        attn_curr = self.attn_drop(self.trans_weights)
        n, nv, nb, h = inputs.shape
        # shape:(batch_size * num_view, num_neighbor, hidden_dim)
        inputs = inputs.reshape(-1, nb, h)
        inputs_trans = torch.matmul(inputs, self.fc)
        # shape:(batch_size * num_view, 1, num_neighbor)
        attention = F.softmax(
            torch.matmul(
                self.tanh(inputs_trans), attn_curr
            ).squeeze(2),
            dim=1
        ).unsqueeze(1)
        return torch.matmul(attention, inputs).squeeze(1).reshape(n, nv, h)


class ViewAttention(Attention):
    def __init__(self, hidden_dim, attn_dim, attn_drop):
        super(ViewAttention, self).__init__(hidden_dim, attn_dim, attn_drop)

    def forward(self, inputs):
        # inputs.shape=(batch_size, num_view, hidden_dim)
        attn_curr = self.attn_drop(self.trans_weights)
        # inputs_trans.shape=(batch_size, num_view, attn_dim)
        inputs_trans = torch.matmul(inputs, self.fc)
        # attention.shape=(batch_size, 1, num_view)
        attention = F.softmax(
            torch.matmul(
                self.tanh(inputs_trans), attn_curr
            ).squeeze(2),
            dim=1
        ).unsqueeze(1)
        # output.shape=(batch_size,hidden_dim)
        return torch.matmul(attention, inputs).squeeze(1)
