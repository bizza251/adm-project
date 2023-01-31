import os
from typing import Callable, Dict, Sequence, Tuple
import torch
from torch.functional import Tensor
from problem_solver import problem_solver
from models.custom_transformer import TSPCustomTransformer
from math import ceil, floor
from random import shuffle
from utility import BatchGraphInput, custom_collate_fn
import queue
from itertools import cycle



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

def map_func(x):
    return BatchGraphInput(x.coords, torch.tensor(x.sub_opt), x.sub_opt_cost)


class RandomGraphDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        path: str,
        mapping_func: Callable[[Dict], BatchGraphInput] = None,
    ):
        super().__init__()
        self.mapping_func = mapping_func if mapping_func is not None else map_func
        self.filenames = [x.path for x in os.scandir(path) if x.is_file()]

    
    def __len__(self):
        return len(self.filenames)


    def __getitem__(self, idx):
        return self.mapping_func(torch.load(self.filenames[idx]))



class RandomGraphDatasetIt(torch.utils.data.IterableDataset):

    def __init__(
        self,
        path: str,
        buffer_size: int = 5000,
        mapping_func: Callable[[Dict], BatchGraphInput] = None,
    ):
        super().__init__()
        self.path = path
        if mapping_func is None:
        #    self.mapping_func = lambda x: BatchGraphInput(*x.values())
            self.mapping_func = map_func#lambda x: BatchGraphInput(x.coords, torch.tensor(x.sub_opt), x.sub_opt_cost)
        else:
            self.mapping_func = mapping_func

        self.buffer_size = buffer_size
        

    def __iter__(self):
        buffer = queue.Queue(self.buffer_size)
        worker_info = torch.utils.data.get_worker_info()
        n_workers = 1 if worker_info is None else worker_info.num_workers
        current_worker_id = 0 if worker_info is None else worker_info.id
        worker_ids = list(range(n_workers))
        for worker_id, item in zip(cycle(worker_ids), os.scandir(self.path)):
            if worker_id == current_worker_id and item.is_file():
                try:
                    buffer.put(item.path, block=False)
                except queue.Full:
                    shuffle(buffer.queue)
                    while not buffer.empty():
                        yield self.mapping_func(torch.load(buffer.get(block=False)))
                    buffer.put(item.path, block=False)
        while not buffer.empty():
            yield self.mapping_func(torch.load(buffer.get(block=False)))
    


def get_dataset(path):
    return RandomGraphDataset(path)


    
def get_dataloader(dataset, batch_size, num_workers, collate_fn=custom_collate_fn):
    return torch.utils.data.DataLoader(
        dataset, 
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn)



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