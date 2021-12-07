import tsplib95
import networkx as nx
import numpy as np

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
    cost = 0
    for i in range(len(path)):
        if (i + 1) == (len(path)):
            break
        cost += G[path[i]][path[i + 1]]["weight"]
    print(f'path found {path} with total cost {cost}')

    print('greedy')

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

    cost_greedy = 0
    for i in range(len(tour_greedy) - 1):
        try:
            cost_greedy += subgraph[(tour_greedy[i],tour_greedy[i+1])]
        except:
            cost_greedy += subgraph[(tour_greedy[i], tour_greedy[0])]
    print(cost_greedy)
    print(np.unique(np.array(tour_greedy).shape))