import torch
from torch import nn


class SigmoidCELoss(nn.Module):
    def __init__(self, num_pos):
        super(SigmoidCELoss, self).__init__()
        self.num_pos = num_pos

    def forward(self, inputs):
        logits = torch.exp(inputs)
        logits = torch.sum(logits[:, :self.num_pos], dim=1) / (torch.sum(logits, dim=1) + 1e-8)
        return -torch.log(logits).mean()