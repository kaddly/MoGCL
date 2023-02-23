import torch
from model.Attention import Attention
import torch.nn.functional as F


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
