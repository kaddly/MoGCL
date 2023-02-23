import torch
from model.Attention import Attention
import torch.nn.functional as F


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

