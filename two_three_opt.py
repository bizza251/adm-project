from typing import List
from torch.functional import Tensor
import torch

from graph import MyGraph


def is_valid_tour(tour: Tensor):
    if not type(tour) is torch.Tensor:
        tour = torch.tensor(tour)
    return tour[0] == tour[-1] and (len(tour) - 1) == len(torch.unique(tour))


def batch_2opt(tour: Tensor, weights: List[dict], n_exchange=3):
    bsz, nodes = tour.shape
    start2end = torch.arange(1, nodes - 1)
    curr_tour = torch.empty_like(tour)
    length = torch.tensor([float('Inf')]).expand(bsz, -1)
    for i in range(n_exchange):
        idx = torch.randperm(nodes - 2)[:2]
        i, k = start2end[idx].sort()[0].chunk(2)
        curr_tour = torch.cat((tour[:, :i], tour[:, i:k + 1].flip(1), tour[:, k + 1:]), dim=1)
        # TODO: compute length of the tours and store them if they're better than current length
        assert nodes == curr_tour.shape[1]
        pass


def batch_3opt(tour: Tensor, weights: List[dict], n_exchange=3):
    bsz, nodes = tour.shape
    start2end = torch.arange(1, nodes - 1)
    curr_tour = torch.empty_like(tour)
    length = torch.tensor([float('Inf')]).expand(bsz, -1)
    for i in range(n_exchange):
        idx = torch.randperm(nodes - 2)[:3]
        i, k, m = start2end[idx].sort()[0].chunk(3)
        curr_tour = torch.cat((tour[:, :i], tour[:, i:k + 1].flip(1), tour[:, k + 1:m + 1].flip(1), tour[:, m + 1:]), dim=1)
        # TODO: compute length of the tours and store them if they're better than current length
        assert nodes == curr_tour.shape[1]
        pass


def _2opt(graph: MyGraph, n_exchange=3):
    try:
        tour = graph.get_opt
    except AttributeError:
        tour = graph.get_sub_opt
    n = len(tour)
    start2end = torch.arange(1, n - 1)
    curr_tour = torch.empty_like(tour)
    best_tour = torch.empty_like(tour)
    length = float('Inf')
    for i in range(n_exchange):
        idx = torch.randperm(n - 2)[:2]
        i, k = start2end[idx].sort()[0].chunk(2)
        curr_tour = torch.cat((tour[:i], tour[i:k + 1].flip(0), tour[k + 1:]), dim=0)
        assert is_valid_tour(tour)
        curr_len = graph.path_cost(tour.tolist())
        if curr_len < length:
            best_tour = curr_tour
            length = curr_len
    return best_tour, length


def _3opt(graph: MyGraph, n_exchange=3):
    try:
        tour = graph.get_opt
    except AttributeError:
        tour = graph.get_sub_opt
    n = len(tour)
    start2end = torch.arange(1, n - 1)
    curr_tour = torch.empty_like(tour)
    best_tour = torch.empty_like(tour)
    length = float('Inf')
    for i in range(n_exchange):
        idx = torch.randperm(n - 2)[:3]
        i, k, m = start2end[idx].sort()[0].chunk(3)
        curr_tour = torch.cat((tour[:i], tour[i:k + 1].flip(0), tour[k + 1:m + 1].flip(0), tour[m + 1:]), dim=0)
        assert is_valid_tour(tour)
        curr_len = graph.path_cost(tour.tolist())
        if curr_len < length:
            best_tour = curr_tour
            length = curr_len        
    return best_tour, length


if __name__ == '__main__':
    bsz, nodes = 3, 10
    tour = torch.stack([torch.randperm(nodes) for _ in range(bsz)])
    tour = torch.cat((tour, tour[:, 0:1]), dim=-1)
    batch_2opt(tour, None, 3)
    batch_3opt(tour, None, 3)