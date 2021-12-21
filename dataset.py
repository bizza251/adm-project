import torch
from torch.functional import Tensor
from problem_solver import problem_solver
from models.custom_transformer import TSPCustomTransformer


class GraphDataset(torch.utils.data.IterableDataset):
    def __init__(self):
        super().__init__()
        self.g = problem_solver()

    def __iter__(self):
        return ((graph.get_n_nodes, graph.coords, graph.get_sub_opt, graph.sub_opt_cost) for graph in self.g)


def gt_matrix_from_tour(tour: Tensor):
    bsz, n = tour.shape
    zero_to_bsz = torch.arange(tour.shape[0]).unsqueeze(1)
    matrix = torch.zeros(bsz, n, n)
    matrix[zero_to_bsz, torch.arange(n), tour] = 1
    return matrix


if __name__ == '__main__':
    dataset = GraphDataset()
    dataloader = torch.utils.data.DataLoader(dataset)
    x = [x for x in dataloader]
    model = TSPCustomTransformer(nhead = 1)
    res = model(x[0][1].to(torch.float32))
    # for i, sample in enumerate(dataloader):
    #     print(sample)
    # tour = torch.stack([item[1] for item in dataloader], dim=0)
    tour = torch.randint(0, 10, (3, 10))
    gt_matrix = gt_matrix_from_tour(tour)
    print(gt_matrix, tour)