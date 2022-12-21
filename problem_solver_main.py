from problem_solver import problem_solver
import torch
import numpy as np
        
if __name__ == '__main__':
    p = problem_solver()
    graphs = [g for g in p]
    mtx_edges = torch.stack([g.get_edges for g in graphs], dim=0)
    print(mtx_edges.shape)