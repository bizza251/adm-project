from gurobipy import *
from gurobipy import multidict, Model, GRB, quicksum
import tsplib95
import numpy as np
import copy


def solve(m, verbose = True):
    m.optimize()
    try:
        if verbose:
            print('Obj: %g' % m.objVal)
            print_sol(m)
    except:
        print('Model infeasible')


def print_sol(m):
    print('---------------')
    print('Solution:')
    for v in m.getVars():
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
        x = m.addVars(n, vtype=GRB.CONTINUOUS, name='city visited')
        for i in range(n):
            x[i] == 0
        arc_used = {}
        for arc in a:
            arc_used[arc] = m.addVar(vtype=GRB.CONTINUOUS, name=f'arc_{arc}_used')
            arc_used[arc] = 0
        flow_in = m.addVars(n, vtype=GRB.CONTINUOUS, name='flow_in')
        for i in range(n):
            flow_in[n] = 0
        flow_in[0] = 1

        #Must visit all edges
        m.addConstr(quicksum(flow_in) == n + 1)
        for i in range(n - 1):
            m.addConstr(flow_in[i] <= 1)
        m.addConstr(flow_in[0] == 2)
        #Flow is transferred in only one edge
        for i in range(start, end):
            m.addConstr(quicksum(arc_used[i, j] for j in range(start, end) if (i,j) in a) == (flow_in[i - 1]))

        for j in range(start, end):
            m.addConstr((quicksum(arc_used[i, j] for i in range(start, end) if (i,j) in a)) == (flow_in[j - 1]))
        


        #OBJECTIVE FUNCTION
        m.setObjective(quicksum(w[arc]*arc_used[arc] for arc in a), GRB.MINIMIZE)
        solve(m, verbose=True)


        ## Devo visitare tutte le città
        #for i in range(n):
        #    m.addConstr(x[i] == 1, name=f'city_{i}_visited')
        ## Da una città posso andare solo in un'altra città
        #for i in range(start, end):
        #    m.addConstr(quicksum(f[i, j] for j in range(start, end) if (
        #        i, j) in a) == 1, name=f'only_one_out_arc')
        ## Posso arrivare in una città solo da una città
        #for j in range(start, end):
        #    m.addConstr(quicksum(f[i, j] for i in range(start, end) if (
        #        i, j) in a) == 1, name=f'only_one_in_arc')
        ## Se visito una città devo arrivarci da un qualche posto
        #for j in range(start, end):
        #    m.addConstr((x[j - 1] == 1) >> (quicksum(f[i, j] for i in range(start,
        #                end) if (i, j) in a) == 1), name=f'visited_city__with_arc')