import torch
from .Attention import Attention


class ViewAttention(Attention):
    def __init__(self, hidden_dim, attn_dim, attn_drop):
        super(ViewAttention, self).__init__(hidden_dim, attn_dim, attn_drop)

    def forward(self, inputs):
        # inputs.shape=(batch_size, num_view, hidden_dim)
        # inputs_trans.shape=(batch_size, num_view, attn_dim)
        inputs_trans = torch.matmul(inputs, self.fc)
        # attention.shape=(batch_size, 1, num_view)
        attention = self.softmax(
            torch.matmul(
                self.tanh(inputs_trans), self.trans_weights
            ).squeeze(2),
            dim=1
        ).unsqueeze(1)
        # output.shape=(batch_size,hidden_dim)
        return torch.matmul(attention, inputs).squeeze(1)
