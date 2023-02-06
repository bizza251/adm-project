from typing import Iterable
import torch
from torch import nn
from torch.functional import Tensor
from utility import get_tour_coords, get_tour_len
from dataclasses import dataclass



@dataclass
class TourModelOutput:
    tour: Tensor
    sum_log_probs: Tensor
    attn_matrix: Tensor = None
        

class TourLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attn_matrix, gt_matrix):
        return torch.mean(torch.square(attn_matrix - gt_matrix))
        # return torch.mean(torch.sum(1 - torch.gather(attn_matrix, 2, gt_tour.unsqueeze(2)).squeeze(), dim=-1))



# class TourLossReinforce(nn.Module):
    
#     def forward(
#         self,
#         coords: Tensor,
#         sum_log_probs: Tensor,
#         tour: Tensor,
#         tgt_len: Tensor,
#         tgt_tour: Tensor = None,
#         attn_matrix: Tensor = None,
#     ) -> Tensor:
#         tour_len = get_tour_len(get_tour_coords(coords, tour))
#         return torch.mean((tour_len - tgt_len) * sum_log_probs)


class TourLossReinforce(nn.Module):
    discount = torch.tensor((0.99), dtype=torch.float).expand(50)
    discount = torch.pow(discount, torch.arange(50)).view(1, 50)
    h_weights = torch.arange(1, 51, dtype=torch.float32)
    h_weights /= h_weights.sum()
    h_weigths = h_weights.unsqueeze(0)
    
    def forward(
        self,
        coords: Tensor,
        sum_log_probs: Tensor,
        tour: Tensor,
        tgt_len: Tensor,
        tgt_tour: Tensor = None,
        attn_matrix: Tensor = None,
    ) -> Tensor:
        bsz, nodes, _ = attn_matrix.shape
        valid_gt_tour = tgt_tour[..., :-1] - 1
        # valid_tour = tour[..., :-1]
        h = (- torch.log(attn_matrix) * attn_matrix).sum(-1)
        h_weigths = self.h_weights.to(attn_matrix.device).expand(bsz, -1)
        h = torch.sum(h * h_weigths, -1).mean()
        valid_tour = torch.distributions.Categorical(attn_matrix).sample().squeeze()
        valid_gt_tour = valid_gt_tour.to(valid_tour.device)
        # reward = torch.where(valid_tour == valid_gt_tour, -1., 1.).contiguous().view(-1).to(attn_matrix.device)
        reward = torch.where(valid_tour == valid_gt_tour, -1., 0.).to(attn_matrix.device)
        # reward = reward.cumsum(1).contiguous().view(-1)
        tour_len = get_tour_len(get_tour_coords(coords, tour))
        reward[tour_len < tgt_len] = -1
        diff = (tour_len - tgt_len).to(attn_matrix.device)
        if torch.any(diff < 0):
            print("Maybe it's possible...")
        reward[tour_len < tgt_len, -1] = -10 + diff[diff < 0]
        reward = reward.cumsum(1).contiguous().view(-1)
        log_probs = torch.gather(attn_matrix, -1, valid_tour.view(bsz, nodes, 1).to(attn_matrix.device)).squeeze().log().contiguous().view(-1)
        return 0.6 * torch.mean(reward * log_probs) + h * 0.4


class TourLossReinforceMixed(nn.Module):
    
    def forward(
        self,
        coords: Tensor,
        sum_log_probs: Tensor,
        tour: Tensor,
        tgt_len: Tensor,
        tgt_tour: Tensor
    ) -> Tensor:
        tour_len = get_tour_len(get_tour_coords(coords, tour))
        if torch.rand(1).item() < 0.99:
            tgt_tour -= 1
            reward = - (tour == tgt_tour).sum(-1).to(sum_log_probs.device)
        else:
            reward = tour_len - tgt_len
        return torch.mean(reward * sum_log_probs)



class ValidTourLossReinforce(nn.Module):

    def __init__(self, penalty=1000):
        super().__init__()
        self.penalty = penalty


    def forward(
        self,
        sum_log_probs: Tensor,
        coords: Tensor,
        tour: Tensor,
        gt_tour: Tensor,
        gt_len: Tensor,
        attn_matrix: Tensor
    ) -> Tensor:
        unique_nodes = torch.tensor([len(set(t.tolist())) for t in tour], dtype=sum_log_probs.dtype, device=sum_log_probs.device)
        expected_unique_nodes = tour.shape[1] - 1
        valid_tour_mask = torch.where(unique_nodes == expected_unique_nodes, True, False)
        rewards = torch.empty((len(sum_log_probs), ), device=sum_log_probs.device)
        if torch.any(valid_tour_mask):
            tour_len = get_tour_len(coords[valid_tour_mask])
            # if torch.rand((1,)).item() < 0.9:
            if False:
                # bsz = tour.shape[0]
                # swap_idxs = torch.randint(1, expected_unique_nodes, (bsz, 2))
                # tmp = coords[torch.arange(bsz), swap_idxs[:, 0]]
                # coords[torch.arange(bsz), swap_idxs[:, 0]] = coords[torch.arange(bsz), swap_idxs[:, 1]]
                # coords[torch.arange(bsz), swap_idxs[:, 1]] = tmp
                coords = coords[:, torch.randperm(coords.shape[1], device=coords.device)]
                bsln_len = get_tour_len(coords)
            else:
                bsln_len = gt_len
            # bsln_len = gt_len
            rewards = tour_len - bsln_len
            # bsln_len = get_tour_len(coords[:, torch.randperm(expected_unique_nodes, device=coords.device)])
            # rewards = tour_len - bsln_len
            # rewards[valid_tour_mask] = tour_len - gt_len[valid_tour_mask]
            # rewards[valid_tour_mask] = gt_len[valid_tour_mask] / tour_len
            # rewards[valid_tour_mask] = torch.where(tour_len > gt_len[valid_tour_mask], tour_len - gt_len[valid_tour_mask], gt_len[valid_tour_mask] - tour_len)
        invalid_tour_mask = ~valid_tour_mask
        if torch.any(invalid_tour_mask):
            # rewards[invalid_tour_mask] = -100
            # rewards[invalid_tour_mask] = torch.tensor(10 * (expected_unique_nodes - unique_nodes[invalid_tour_mask]), \
            #     dtype=sum_probs.dtype, device=sum_probs.device)
            # rewards[invalid_tour_mask] = (unique_nodes[invalid_tour_mask] / expected_unique_nodes)
            rewards[invalid_tour_mask] = (tour[invalid_tour_mask] != gt_tour[invalid_tour_mask]).sum(-1)
        # h_loss = (- torch.log(attn_matrix) * attn_matrix).sum(1).mean(-1).mean()
        return torch.mean(rewards * sum_log_probs)



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