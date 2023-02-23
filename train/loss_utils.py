import torch
from torch import nn


class SigmoidBCELoss(nn.Module):
    def __init__(self, num_pos, num_nodes, device):
        super(SigmoidBCELoss, self).__init__()
        self.gt = torch.concat([torch.ones(num_pos, dtype=torch.long, device=device),
                                torch.zeros(num_nodes - num_pos, dtype=torch.long, device=device)])

    def forward(self, inputs):
        out = nn.functional.binary_cross_entropy_with_logits(inputs, self.gt, reduction="none")
        return out.mean(dim=1)
