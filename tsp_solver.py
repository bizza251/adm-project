from gurobipy import *
from gurobipy import multidict, Model, GRB, quicksum
import tsplib95
import numpy as np
import copy

def solve(m):
    m.optimize()
    try:
        print('Obj: %g' % m.objVal)
        print_sol(m)
    except:
        print('Model infeasible')


def print_sol(m):
    print('---------------')
    print('Solution:')
    for v in m.getVars():
        if v.x != 0:
            print('%s %g' % (v.varName, v.x))


if __name__ == '__main__':
    problem_path = ['ALL_tsp/a280.tsp']
    n = 50

    problems = [tsplib95.load(path) for path in problem_path]

    dizs = []
    for i, problem in enumerate(problems):
        G = problem.get_graph()
        subgraph_edges = np.array(list(G.nodes))[:50]
        G = G.subgraph(subgraph_edges)
        dizs.append({edge : G.get_edge_data(edge[0], edge[1])['weight'] for edge in G.edges if edge[0] != edge[1]})
    
    
    start = 1
    end = n + 1
    for diz in dizs:
        a, w = multidict(diz)
        m = Model('TSP_solver')
        x = m.addVars(n, vtype=GRB.BINARY, name='city visited')
        f = {}
        for arc in a:
            #print(arc)
            f[arc] = m.addVar(vtype=GRB.BINARY, name=f'arc_{arc}_taken')
        m.addConstr(quicksum(x) == n)
        #Da una città posso andare solo in un'altra città
        for i in range(start, end):
            m.addConstr(quicksum(f[i,j] for j in range(start, end) if (i,j) in a) == 1, name=f'only_one_out_arc')
        #Posso arrivare in una città solo da una città
        for j in range(start, end):
            m.addConstr(quicksum(f[i,j] for i in range(start, end) if (i,j) in a) == 1, name=f'only_one_in_arc')
        #
        for j in range(start, end):
            m.addConstr(quicksum(f[i,j] for i in range(start, end) if (i,j) in a) <= x[j - 1] * GRB.MAXINT, name=f'visited_city__with_arc')
        m.setObjective(quicksum(w[arc]*f[arc] for arc in a), GRB.MINIMIZE)
        m.optimize()