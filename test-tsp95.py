import tsplib95
import torch

from torch.nn.modules.transformer import Transformer

GRAPH_PATH = 'ALL_tsp/fl3795.tsp'

problem = tsplib95.load(GRAPH_PATH) 

pass