from re import sub
import tsplib95
import networkx as nx
import numpy as np
from utility import path_cost
from graph import MyGraph

problem_path = ['ALL_tsp/bayg29.tsp']

problems = [tsplib95.load(path) for path in problem_path]
n = 29

dizs = []
start = 1
end = n + 1

for i, problem in enumerate(problems):
    nodes = [i for i in range(start, end)]
    edges = [(i, j) for i in nodes for j in nodes if i != j]
    subgraph = {edge: problem.get_weight(
        edge[0], edge[1]) for edge in edges}


    tsp = nx.approximation.traveling_salesman_problem
    G = nx.complete_graph(n)

    for edge in edges:
        G[edge[0] - 1][edge[1] -1 ]["weight"] = subgraph[edge]
    path = tsp(G, cycle=True)
    path = np.array(path) + 1
    cost = 0
    cost = path_cost(path, subgraph)
    print(f'NX algorithm: Path found {list(path)} with total cost {cost} and lenght {len(path)}')

    tour_greedy = []
    tour_greedy.append(nodes[0])

    while len(tour_greedy) < n:
        cur_edges = []
        cur_weights = []
        cur_node = tour_greedy[-1]
        for edge in edges:
            if edge[0] == cur_node:
                cur_edges.append(edge[1])
                cur_weights.append(subgraph[edge])
                #print(f'{edge} weight {subgraph[edge]}')
        cur_edges = np.array(cur_edges)
        cur_weights = np.array(cur_weights)
        node = None
        while node is None:
            min_idx = np.argmin(cur_weights)
            if cur_edges[min_idx] in tour_greedy:
                cur_edges = np.delete(cur_edges, min_idx)
                cur_weights = np.delete(cur_weights, min_idx)
                continue
            node = cur_edges[min_idx]
        tour_greedy.append(node)
    tour_greedy.append(nodes[0])
    print(tour_greedy, len(tour_greedy))

    cost_greedy = path_cost(tour_greedy, subgraph)  

    print(f'Greedy algorithm: {tour_greedy} with cost {cost_greedy} and lenght {len(tour_greedy)}')

    my_g = MyGraph(problem_path[i], nodes = nodes, weights = subgraph, sub_opt = tour_greedy)
    print(my_g.path_cost(my_g.sub_opt))
    print(my_g.name)