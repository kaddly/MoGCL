import torch
from .Attention import Attention


class NeighborEncoder(Attention):
    def __init__(self, hidden_dim, attn_dim, attn_drop):
        super(NeighborEncoder, self).__init__(hidden_dim, attn_dim, attn_drop)

    def forward(self, inputs):
        # inputs.shape:(batch_size, num_view, num_neighbor, hidden_dim)
        n, nv, nb, h = inputs.shape
        # shape:(batch_size * num_view, num_neighbor, hidden_dim)
        inputs = inputs.reshape(-1, nb, h)
        inputs_trans = torch.matmul(inputs, self.fc)
        # shape:(batch_size * num_view, 1, num_neighbor)
        attention = self.softmax(
            torch.matmul(
                self.tanh(inputs_trans), self.trans_weights
            ).squeeze(2),
            dim=1
        ).unsqueeze(1)
        return torch.matmul(attention, inputs).reshape(n, nv, h)

