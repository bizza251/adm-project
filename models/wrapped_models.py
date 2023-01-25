from typing import OrderedDict
from torch import Tensor
import torch
import torch.nn as nn
from models.utility import TourModelOutput
from dataclasses import dataclass
import copy



@dataclass
class TourModelWithBaselineOutput(TourModelOutput):
    bsln: TourModelOutput = None



class RLAgentWithBaseline(nn.Module):

    def __init__(
        self,
        model: nn.Module
    ) -> None:
        super().__init__()
        self.model = model
        self.bsln = copy.deepcopy(model)
        self.bsln.eval()

    
    def train(self, mode: bool = True):
        self.model.train(mode)


    def eval(self):
        self.model.eval()

    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict(destination, prefix, keep_vars)

    
    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                    strict: bool = True):
        self.model.load_state_dict(state_dict, strict)

    
    def update_bsln(self, state_dict: 'OrderedDict[str, Tensor]' = None):
        if state_dict is None:
            state_dict = self.model.state_dict()
        self.bsln.load_state_dict(state_dict)
    
    
    def forward(self, x, *args, **kwargs):
        model_out = self.model(x, *args, **kwargs)
        # if self.model.training:
        with torch.no_grad():
            bsln_out = self.bsln(x, *args, **kwargs)
            return TourModelWithBaselineOutput(**vars(model_out), bsln=bsln_out)
        # return TourModelWithBaselineOutput(**vars(model_out))
