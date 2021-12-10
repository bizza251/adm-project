from typing import List
from torch.functional import Tensor
import torch


def _2opt(tour: Tensor, weights: List[dict], n_exchange=3):
    bsz, nodes = tour.shape
    start2end = torch.arange(1, nodes - 1)
    new_tour = torch.empty_like(tour)
    length = torch.tensor([float('Inf')]).expand(bsz, -1)
    for i in range(n_exchange):
        idx = torch.randperm(nodes - 2)[:2]
        i, k = start2end[idx].sort()[0].chunk(2)
        new_tour_ = torch.cat((tour[:, :i], tour[:, i:k + 1].flip(1), tour[:, k + 1:]), dim=1)
        # TODO: compute length of the tours and store them if they're better than current length
        assert nodes == new_tour_.shape[1]
        pass


def _3opt(tour: Tensor, weights: List[dict], n_exchange=3):
    bsz, nodes = tour.shape
    start2end = torch.arange(1, nodes - 1)
    new_tour = torch.empty_like(tour)
    length = torch.tensor([float('Inf')]).expand(bsz, -1)
    for i in range(n_exchange):
        idx = torch.randperm(nodes - 2)[:3]
        i, k, m = start2end[idx].sort()[0].chunk(3)
        new_tour_ = torch.cat((tour[:, :i], tour[:, i:k + 1].flip(1), tour[:, k + 1:m + 1].flip(1), tour[:, m + 1:]), dim=1)
        # TODO: compute length of the tours and store them if they're better than current length
        assert nodes == new_tour_.shape[1]
        pass



if __name__ == '__main__':
    bsz, nodes = 3, 10
    tour = torch.stack([torch.randperm(nodes) for _ in range(bsz)])
    tour = torch.cat((tour, tour[:, 0:1]), dim=-1)
    _2opt(tour, None, 3)
    _3opt(tour, None, 3)