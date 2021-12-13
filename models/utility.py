import torch
from torch import nn


class TourLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attn_matrix, gt_tour):
        return torch.mean(torch.sum(1 - torch.gather(attn_matrix, 2, gt_tour.unsqueeze(2)).squeeze(), dim=-1))