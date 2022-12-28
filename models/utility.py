from typing import Iterable
import torch
from torch import nn
from torch.functional import Tensor



def get_tour_len(tour: Tensor) -> Tensor:
    """Compute the length of a batch of tours.

    Args:
        tour (Tensor): shape (N, L, D)

    Returns:
        Tensor: shape (N), contains the length of each tour in the batch.
    """   
    bsz, nodes, features = tour.shape 
    diff = torch.diff(tour, dim=1, append=torch.zeros(bsz, 1, features))
    return diff.square().sum(dim=-1).sqrt().sum(dim=-1)

        

class TourLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attn_matrix, gt_tour):
        return torch.mean(torch.sum(1 - torch.gather(attn_matrix, 2, gt_tour.unsqueeze(2)).squeeze(), dim=-1))



class TourLossReinforce(nn.Module):
    
    def forward(
        self,
        sum_log_probs: Tensor,
        tour: Tensor,
        gt_len: Tensor
    ) -> Tensor:
        tour_len = get_tour_len(tour)
        return torch.mean((tour_len - gt_len) * sum_log_probs)



def get_node_mask(n: int, to_mask: Iterable[Tensor]):
    """Generate an additive attention mask to force the model to select a given node in a given position.

    Args:
        n (int): number of nodes in the graph
        to_mask Iterable[Tensor]: if it's a tensor, then the shape must be (*, 2), where * can be any number of nodes 
                        and 2 is a pair of indeces representing the position of the node in the tour and the node itself.
                        If it's not a tensor, it must be an iterable of pairs of indices.
    """ 
    mask = torch.zeros((n, n))
    if type(to_mask) is torch.Tensor:
        mask[to_mask[:, 0]] = float('-Inf')
        mask[:, to_mask[:, 1]] = float('-Inf')
        mask[to_mask[:, 0], to_mask[:, 1]] = 0
    else:
        for pos in to_mask:
            mask[pos[0]] = float('-Inf')
            mask[:, pos[1]] = float('-Inf')
            mask[pos[0], pos[1]] = 0
    return mask


if __name__ == '__main__':
    to_mask = ((3, 9), (5, 1))
    mask = get_node_mask(10, to_mask)
    print(mask)
    to_mask = torch.tensor(to_mask)
    mask = get_node_mask(10, to_mask)
    print(mask)