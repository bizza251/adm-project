from torch import nn
from torch.functional import Tensor
from constants import *
import torch
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class GraphEncoderTransformer(nn.Module):
    def __init__(self, d_model=ENCODER_D_MODEL, 
            num_layers=ENCODER_LAYERS,
            nhead=8,
            dropout_p=0.2,
            activation=nn.functional.relu) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout_p = dropout_p
        self.activation = activation
        encoder_layer = nn.modules.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout_p, batch_first=True, norm_first=True, activation=self.activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.modules.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers, norm=encoder_norm)
        self.embd = nn.Embedding(MAX_LEN, d_model)
        self.pos_enc = PositionalEncoding(d_model=d_model, dropout=dropout_p, max_len=MAX_LEN)
        mask = nn.parameter.Parameter(torch.rand(d_model))
        self.register_buffer('mask', mask)

    def mask(self, x: Tensor, idxs: Tensor):
        x[idxs[idxs > 0]] = self.mask
        return x
    
    def forward(self, x):
        # x has shape (N, L, D) 
        # N -> batch size
        # L -> number of nodes
        # D -> embedding dim
        x = 