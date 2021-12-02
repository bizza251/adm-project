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
    start = 1
    end = n + 1
    for i, problem in enumerate(problems):
        nodes = [i for i in range(start, end)]
        edges = [(i, j) for i in nodes for j in nodes if i != j]
        subgraph = {edge: problem.get_weight(
            edge[0], edge[1]) for edge in edges}
        dizs.append(subgraph)

    for diz in dizs:
        a, w = multidict(diz)
        m = Model('TSP_solver'
                  )
        x = m.addVars(n, vtype=GRB.BINARY, name='city visited')
        f = {}
        for arc in a:
            f[arc] = m.addVar(vtype=GRB.BINARY, name=f'arc_{arc}_used')
        # Devo visitare tutte le città
        for i in range(n):
            m.addConstr(x[i] == 1, name=f'city_{i}_visited')
        # Da una città posso andare solo in un'altra città
        for i in range(start, end):
            m.addConstr(quicksum(f[i, j] for j in range(start, end) if (
                i, j) in a) == 1, name=f'only_one_out_arc')
        # Posso arrivare in una città solo da una città
        for j in range(start, end):
            m.addConstr(quicksum(f[i, j] for i in range(start, end) if (
                i, j) in a) == 1, name=f'only_one_in_arc')
        # Se visito una città devo arrivarci da un qualche posto
        for j in range(start, end):
            m.addConstr((x[j - 1] == 1) >> (quicksum(f[i, j] for i in range(start,
                        end) if (i, j) in a) == 1), name=f'visited_city__with_arc')
        # TODO aggiungere vincolo che da un dato nodo devo poter raggiungere tutti gli altri, attenzione anche che devo poter percorrerli al contrario
        for arc in a:
            if f[arc] == 1 and (arc[1], arc[0]) in a:
                    f[arc[1],arc[0]] = 1

        m.setObjective(quicksum(w[arc]*f[arc] for arc in a), GRB.MINIMIZE)
        solve(m)
