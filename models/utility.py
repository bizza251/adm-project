from typing import Iterable
import torch
from torch import nn
from torch.functional import Tensor
from utility import get_tour_len

        

class TourLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attn_matrix, gt_tour):
        return torch.mean(torch.sum(1 - torch.gather(attn_matrix, 2, gt_tour.unsqueeze(2)).squeeze(), dim=-1))



class TourLossReinforce(nn.Module):
    
    def forward(
        self,
        sum_probs: Tensor,
        coords: Tensor,
        gt_len: Tensor
    ) -> Tensor:
        tour_len = get_tour_len(coords)
        return torch.mean((tour_len - gt_len) * torch.log(sum_probs))



class ValidTourLossReinforce(nn.Module):

    def __init__(self, penalty=1000):
        super().__init__()
        self.penalty = penalty


    def forward(
        self,
        sum_probs: Tensor,
        coords: Tensor,
        tour: Tensor,
        gt_len: Tensor
    ) -> Tensor:
        unique_nodes = torch.tensor([len(set(t.tolist())) for t in tour], dtype=sum_probs.dtype, device=sum_probs.device)
        expected_unique_nodes = tour.shape[1] - 1
        valid_tour_mask = torch.where(unique_nodes == expected_unique_nodes, True, False)
        rewards = torch.empty((len(sum_probs), ), device=sum_probs.device)
        if torch.any(valid_tour_mask):
            tour_len = get_tour_len(coords[valid_tour_mask])
            rewards[valid_tour_mask] = (tour_len - gt_len[valid_tour_mask])
        invalid_tour_mask = ~valid_tour_mask
        if torch.any(invalid_tour_mask):
            rewards[invalid_tour_mask] = self.penalty
        return torch.mean(rewards * torch.log(sum_probs))



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