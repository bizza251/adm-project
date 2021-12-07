import tsplib95
import networkx as nx

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
    dizs.append(subgraph)


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