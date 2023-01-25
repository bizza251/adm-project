from typing import Sequence
import numpy as np
from scipy.spatial.distance import pdist
import networkx as nx
from torch import Tensor
from dataclasses import dataclass
import torch
import os


def path_cost(path : list, weights : dict, cycle=True) -> int:
    """Computes the cost of the given path

    Args:
        path (list): tour taken\n
        weights (dict): key -> edge : value -> weight.\n
        cycle (bool, optional): [description]. Defaults to True.\n

    Returns:
        int: final cost
    """
    cost = 0
    for i in range(len(path) - 1):
        try:
            if (path[i], path[i + 1]) in weights.keys():
                cost += weights[(path[i],path[i+1])]
            else:
                cost += weights[(path[i + 1],path[i])]
        except:
            if cycle:
                if (path[i], path[0]) in weights.keys():
                    cost += weights[(path[i], path[0])]
                else:
                    cost += weights[(path[0], path[i])]
                pass
    return cost


def read_file_from_directory(path, type = None, absolute=False):
    """Return all files from the given directory

    Args:
        path ([type]): [description]
        type ([type], optional): [Return only the file with the given extension]. Defaults to None.
        absolute (bool, optional): [Returns a dictionary instead of a list with keys the filename and values the absolute path]. Defaults to False.

    Returns:
        [type]: [description]
    """
    import os
    if not absolute:
        filename = []
        for file in os.listdir(path):
            if type != None and file.split('.')[-1] == type:
                filename.append(file)
    else:
        filename = {file : os.path.join(path, file) for file in os.listdir(path) if type != None and file.split('.')[-1] == type}
    return filename

def create_dir(path):
    import os
    try:
        os.mkdir(path)
        print('Path created correctly')
    except:
        pass



def random_tsp_instance(n, features=2, max_norm=True, low=None, high=None):
    if low is not None:
        nodes = np.random.randint(low, high, (n, features))
    else:
        nodes = np.random.rand(n ,features)
    if max_norm:
        nodes /= np.absolute(nodes.max(axis=0))
    if nodes.dtype == np.float64:
        nodes = nodes.astype(np.float32)
    G = nx.complete_graph(n)
    edge_keys = [e for e in G.edges()]
    nx.set_node_attributes(G, {k: v for k, v in zip(np.arange(n), nodes)}, 'pos')
    nx.set_edge_attributes(G, {k: v for k, v in zip(edge_keys, pdist(nodes))}, 'dist')
    return G, nodes


def random_tsp_instance_solved(n, features=2, max_norm=True, low=None, high=None):
    G, nodes = random_tsp_instance(n, features, max_norm, low, high)
    tour = nx.approximation.traveling_salesman_problem(G, cycle=True)
    length = 0
    for i in range(n):
        length += G[tour[i]][tour[i + 1]]['dist']
    return G, nodes, tour, length



def create_random_dataset(path, n, n_nodes, n_features, max_norm=True, n_processes=4):
    from multiprocessing.pool import Pool
    from multiprocessing import Value
    import pathlib
    from functools import partial
    from uuid import uuid4
    from sys import stdout
    import ctypes
    from random import random

    pathlib.Path(path).mkdir(parents=True, exist_ok=True) 

    completed = Value(ctypes.c_ulong, 0)

    global save_sample
    def save_sample(_, n_nodes, n_features, max_norm=True):
        _, nodes, tour, length = random_tsp_instance_solved(n_nodes, n_features, max_norm)
        torch.save(dict(
            nodes=torch.tensor(nodes),
            tour=torch.tensor(tour),
            length=torch.tensor(length)
        ),
        os.path.join(path, f"{uuid4()}.pt"))
        completed.value += 1
        if random() > 0.99:
            with completed.get_lock():
                i = completed.value
                stdout.write(f"[{i}/{n}] done ({(i / n * 100):.2f}%)\n")

    func = partial(save_sample, n_nodes=n_nodes, n_features=n_features, max_norm=max_norm)

    with Pool(n_processes) as P:
        P.map(func, range(n), chunksize=n // n_processes)        
    del save_sample
    
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logging.basicConfig(
        format="%(message)s"
        )   
    logger.info('Done!')



@dataclass
class BatchGraphInput:
    coords: Tensor
    gt_tour: Tensor
    gt_len: float

    def __len__(self):
        return 1 if len(self.coords.shape) < 3 else len(self.coords)



def custom_collate_fn(samples: Sequence[BatchGraphInput]):
    return BatchGraphInput(
            torch.stack([sample.coords for sample in samples]),
            torch.stack([sample.gt_tour for sample in samples]),
            torch.tensor([sample.gt_len for sample in samples], dtype=torch.float32)
    )
    # try:
    #     return BatchGraphInput(
    #         torch.stack([sample.coords for sample in samples]),
    #         torch.stack([sample.gt_tour for sample in samples]),
    #         torch.tensor([sample.gt_len for sample in samples], dtype=torch.float32)
    #     )
    # except TypeError:
    #     return BatchGraphInput(
    #         torch.stack([torch.tensor(sample.coords) for sample in samples]),
    #         torch.stack([torch.tensor(sample.gt_tour) for sample in samples]),
    #         torch.tensor([sample.gt_len for sample in samples], dtype=torch.float32)
    #     )


def get_tour_len(tour: Tensor) -> Tensor:
    """Compute the length of a batch of tours.

    Args:
        tour (Tensor): shape (N, L, D)

    Returns:
        Tensor: shape (N), contains the length of each tour in the batch.
    """   
    bsz, _, features = tour.shape 
    diff = torch.diff(tour, dim=1)
    return diff.square().sum(dim=-1).sqrt().sum(dim=-1)



def get_tour_coords(coords, tour):
    return coords[torch.arange(len(tour)).view(-1, 1), tour]



def len_to_gt_len_ratio(model_output, batch):
    tours = model_output.tour
    tour_coords = batch.coords[torch.arange(len(tours)).view(-1, 1), tours]
    tour_len = get_tour_len(tour_coords)
    return (tour_len.cpu() / batch.gt_len).mean().item()



def valid_tour_ratio(model_output, batch):
    tours = model_output.tour
    expected_unique_nodes = tours.shape[1] - 1
    unique_nodes = torch.tensor([len(set(x.tolist())) for x in tours])
    return ((unique_nodes == expected_unique_nodes).sum() / tours.shape[0]).item()     



def avg_tour_len(model_output, batch):          
    tours = model_output.tour
    tour_coords = batch.coords[torch.arange(len(tours)).view(-1, 1), tours]
    tour_len = get_tour_len(tour_coords)
    return tour_len.mean().item()



if __name__ == '__main__':
    create_random_dataset('ALL_tsp/random/train_debug', int(1e2), 50, 2)