import tsplib95
import networkx as nx
import numpy as np
from utility import path_cost
from graph import MyGraph
import os
import torch
from utility import read_file_from_directory


def problem_solver(path : str = os.path.join('.','ALL_tsp','Uncompressed'),  min_n : int = 50, verbose : bool = False):
    """Returns a generator of all the solved problems that satisfy the minumum number of edges.


    Args:
        path (str, optional): [Absolute path to the folder containing the problem files]. Defaults to os.path.join('.','ALL_tsp').
        min_n (int, optional): [Minimum and maximum number of nodes to consider]. Defaults to 50.
        verbose (bool, optional): Defaults to False.

    Yields:
        [type]: [description]
    """
    problem_path = read_file_from_directory(path, 'tsp', absolute=True)
    problems = [tsplib95.load(problem_path[path]) for path in problem_path]

    for i, problem in enumerate(problems):
        n = len(list(problem.get_nodes()))
        if n < min_n:
            if verbose: print(f'Scarto perchè numero di nodi inferiore a {min_n}')
            continue
        else:
            n = min_n
        start = 1
        end = n + 1
        nodes = [i for i in range(start, end)]
        edges = [(i, j) for i in nodes for j in nodes if i != j]
        subgraph = {edge: problem.get_weight(
            edge[0], edge[1]) for edge in edges}

        coords = problem.node_coords
        tmp = []
        for i in range(50):
            tmp.append(coords[i+1])
        coords = torch.tensor(tmp)

        tsp = nx.approximation.traveling_salesman_problem
        G = nx.complete_graph(n)

        for edge in edges:
            G[edge[0] - 1][edge[1] -1 ]["weight"] = subgraph[edge]
        path = tsp(G, cycle=True)
        if len(path) != min_n + 1:
            continue
        path = np.array(path) + 1
        cost = 0
        cost = path_cost(path, subgraph)
        if verbose: print(f'NX algorithm: Path found {list(path)} with total cost {cost} and lenght {len(path)}')

        #tour_greedy = []
        #tour_greedy.append(nodes[0])
        #while len(tour_greedy) < n:
        #    cur_edges = []
        #    cur_weights = []
        #    cur_node = tour_greedy[-1]
        #    for edge in edges:
        #        if edge[0] == cur_node:
        #            cur_edges.append(edge[1])
        #            cur_weights.append(subgraph[edge])
        #    cur_edges = np.array(cur_edges)
        #    cur_weights = np.array(cur_weights)
        #    node = None
        #    while node is None:
        #        min_idx = np.argmin(cur_weights)
        #        if cur_edges[min_idx] in tour_greedy:
        #            cur_edges = np.delete(cur_edges, min_idx)
        #            cur_weights = np.delete(cur_weights, min_idx)
        #            continue
        #        node = cur_edges[min_idx]
        #    tour_greedy.append(node)
        #tour_greedy.append(nodes[0])
        #if verbose: print(tour_greedy, len(tour_greedy))
        #cost_greedy = path_cost(tour_greedy, subgraph)  
        #if verbose: print(f'Greedy algorithm: {tour_greedy} with cost {cost_greedy} and lenght {len(tour_greedy)}')
        if problem.name.split('.')[-1] != 'tsp':
            yield MyGraph(problem_path[problem.name + '.tsp'], nodes = nodes, coords = coords,weights = subgraph, sub_opt = path, sub_opt_cost = cost)
        else: continue
            
if __name__ == '__main__':
    p = problem_solver(verbose = False)
    g = next(p)
    with open('test' + '.txt', 'w') as outfile:
        outfile.write(str(g.__dict__))
        #print(str(g.__dict__))
        outfile.close()
    with open('test' + '.txt', 'r') as infile:
        tmp = infile.read()
        print(tmp)
    pass