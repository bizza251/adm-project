import os
from typing import Callable, Dict, Sequence, Tuple
import torch
from torch.functional import Tensor
from problem_solver import problem_solver
from models.custom_transformer import TSPCustomTransformer
from math import ceil, floor

from utility import BatchGraphInput


def split_data(data, train_p, val_p, test_p):
    assert train_p + val_p + test_p + 1e-8 >= 1, "Percentages must sum to 1"
    n = len(data)
    train_n = ceil(n * train_p)
    val_n = floor(n * val_p)
    train, val, test = data[:train_n], data[train_n:train_n + val_n], data[val_n:]
    assert len(train) + len(val) + len(test) == n, "Error in splitting"
    return train, val, test
    

class GraphDataset(torch.utils.data.IterableDataset):
    def __init__(self):
        super().__init__()
        self.g = problem_solver()

    def __iter__(self):
        # out  = ((graph.get_n_nodes, graph.coords, graph.get_sub_opt, graph.sub_opt_cost) for graph in self.g)
        out = (
            BatchGraphInput(
                graph.coords,
                graph.get_sub_opt,
                graph.sub_opt_cost
            ) for graph in self.g
        )
        self.g = problem_solver()
        return out


def gt_matrix_from_tour(tour: Tensor):
    bsz, n = tour.shape
    zero_to_bsz = torch.arange(bsz).unsqueeze(1)
    matrix = torch.zeros(bsz, n, n)
    matrix[zero_to_bsz, torch.arange(n), tour] = 1
    return matrix



class RandomGraphDataset(torch.utils.data.IterableDataset):

    def rewind(self):
        def get_generator():
            for item in os.scandir(self.path):
                if item.is_file() and item.name.endswith('.pt'):
                    yield item
        self.g = get_generator()


    def __init__(
        self,
        path: str,
        mapping_func: Callable[[Dict], BatchGraphInput] = None,
    ):
        self.path = path
        if mapping_func is None:
           self.mapping_func = lambda x: BatchGraphInput(*x.values())
        else:
            self.mapping_func = mapping_func

        self.rewind()


    def __iter__(self):
        out = (self.mapping_func(torch.load(x.path)) for x in self.g)
        self.rewind()
        return out
    


if __name__ == '__main__':
    dataset = GraphDataset()
    dataloader = torch.utils.data.DataLoader(dataset)
    x = [x for x in dataloader]
    model = TSPCustomTransformer(nhead = 1)
    res = model(x[0][1].to(torch.float32))
    # for i, sample in enumerate(dataloader):
    #     print(sample)
    # tour = torch.stack([item[1] for item in dataloader], dim=0)
    tour = torch.randint(0, 10, (3, 10))
    gt_matrix = gt_matrix_from_tour(tour)
    print(gt_matrix, tour)