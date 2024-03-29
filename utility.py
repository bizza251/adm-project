import numpy as np
from scipy.spatial.distance import pdist
import networkx as nx
from torch import Tensor
from dataclasses import dataclass
import torch
import os
import logging
import torch
import numpy as np
from typing import Union, Callable, Sequence



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(levelname)s: %(message)s"
)



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
            if (path[i].item(), path[i + 1].item()) in weights.keys():
                cost += weights[(path[i].item(),path[i+1].item())]
            else:
                cost += weights[(path[i + 1].item(),path[i].item())]
        except:
            if cycle:
                if (path[i].item(), path[0].item()) in weights.keys():
                    cost += weights[(path[i].item(), path[0].item())]
                else:
                    cost += weights[(path[0].item(), path[i].item())]
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
    gt_len: Union[float, Tensor]
    id: Union[str, Sequence[str]]= None

    def __len__(self):
        return 1 if len(self.coords.shape) < 3 else len(self.coords)



def custom_collate_fn(samples: Sequence[BatchGraphInput]):
    return BatchGraphInput(
            torch.stack([sample.coords for sample in samples]),
            torch.stack([sample.gt_tour for sample in samples]),
            torch.tensor([sample.gt_len for sample in samples], dtype=torch.float32),
            [sample.id for sample in samples]
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


def get_tour_coords(coords, tour):
    return coords[torch.arange(len(tour)).view(-1, 1), tour]



def get_tour_len(coords: Tensor, tour: Tensor = None) -> Tensor:
    """Compute the length of a batch of tours.

    Args:
        tour (Tensor): shape (N, L, D)

    Returns:
        Tensor: shape (N), contains the length of each tour in the batch.
    """   
    if tour is not None:
        coords = get_tour_coords(coords, tour)
    diff = torch.diff(coords, dim=1)
    return diff.square().sum(dim=-1).sqrt().sum(dim=-1)



# def perturb(tour: np.array, n_permutations: int = 2) -> np.array:
#     for i in range(n_permutations):
#         x, y = None, None
#         while (x is None or y is None) or (x == y):
#             x, y = np.random.randint(1, tour.shape[0]-1), np.random.randint(1, tour.shape[0]-1)
#         _ = tour[x]
#         tour[x] = tour[y]
#         tour[y] = _
#     return tour


def perturb(tour: np.array, n_permutations: int = 2) -> np.array:
    tour = np.array(tour)
    for i in range(n_permutations):
        x, y = 0, 0
        while (x == y):
            x, y = np.random.randint(1, tour.shape[0]-1), np.random.randint(1, tour.shape[0]-1)
        if x == 0 or x == tour.shape[-1]:
            print('x is trying to perturb start or end of tour')
        if y == 0 or y == tour.shape[-1]:
            print('y is trying to perturb start or end of tour')
        _ = tour[x]
        tour[x] = tour[y]
        tour[y] = _
    return tour



def hillclimbing(objective: Callable, graph, start_pt: np.array, n_iterations: int, n_permutations=15) -> Union[np.array, float]:
    best = start_pt
    best_eval = objective(best, graph.weights)
    for i in range(n_iterations):
        start_pt = None
        while start_pt is None:
            start_pt = perturb(best, n_permutations)
            proposed_tour, proposed_eval = start_pt, objective(start_pt, graph.weights)
        if proposed_eval < best_eval:
            best, best_eval = proposed_tour, proposed_eval
    return best, best_eval


def iterated_local_search(objective: Callable, graph, n_restarts: int, n_iterations: int, start_pt: np.array, 
    n_permutations=30,
    n_permutations_hillclimbing=30) -> Union[np.array, float]:

    best = start_pt
    best_eval = objective(best, graph.weights)
    for i in range(n_restarts):
        start_pt = None
        while start_pt is None:
            start_pt = perturb(best, n_permutations)
        proposed_tour, proposed_eval = hillclimbing(objective, graph, start_pt, n_iterations, n_permutations_hillclimbing)
        if proposed_eval < best_eval:
            best, best_eval = proposed_tour, proposed_eval
    return best, best_eval



def np2nx(x: np.ndarray):
    G = nx.Graph()
    for i, node in enumerate(x):
        G.add_node(i + 1, pos=node)
        for j, node2 in enumerate(x):
            if i != j:
                d = ((node - node2) ** 2).sum() ** 0.5
                G.add_edge(i + 1, j + 1, weight=d)
    return G



if __name__ == '__main__':
    create_random_dataset('ALL_tsp/random/train_debug', int(1e2), 50, 2)