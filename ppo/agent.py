import torch
import torch.nn as nn
from models.custom_transformer import TSPCustomTransformer
from torch import Tensor
import numpy as np



@torch.no_grad()
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    if type(layer) is nn.Linear:
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer



class PPOAgent(nn.Module):
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.apply(layer_init)

    
    def forward(self, x: Tensor, *args, **kwargs):
        return self.model(x, *args, **kwargs)