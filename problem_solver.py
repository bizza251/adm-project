import torch
from utility import read_file_from_directory


def problem_solver(path: str = 'our_graphs'):
    filename = read_file_from_directory(path, absolute=True, type='txt')
    for file in filename:
        yield torch.load(filename[file])
    
if __name__ == '__main__':
    gen = problem_solver()
    