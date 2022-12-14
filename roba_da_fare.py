import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
from random import random


def ER(n):
    V = set([v + 1 for v in range(n)])
    E = set()
    for combination in combinations(V, 2):
        E.add(combination)

    g = nx.Graph()
    g.add_nodes_from(V)
    g.add_edges_from(E)

    return g


n = 50
G = ER(n)
pos = nx.spring_layout(G)
for edge in G.edges:
    
    #G[edge[0]][edge[1]]['weight'] = 1
    print(G[edge[0]][edge[1]])
        
#nx.draw_networkx(G, pos)
#plt.title("Random Graph Generation Example")
#plt.show()
