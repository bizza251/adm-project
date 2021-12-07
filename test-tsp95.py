import tsplib95

GRAPH_PATH = 'ALL_tsp/bayg29.tsp'

problem = tsplib95.load(GRAPH_PATH)

opt = tsplib95.load('ALL_tsp/bayg29.opt.tour')
print(problem.trace_tours(opt.tours))

pass