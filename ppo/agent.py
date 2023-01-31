import torch
import torch.nn as nn
from models.custom_transformer import TSPCustomTransformer
from torch import Tensor



class PPOAgent(nn.Module):
    
    def __init__(self, model):
        super().__init__()
        self.model = model

    
    def forward(self, x: Tensor, *args, **kwargs):
        return self.model(x, *args, **kwargs)