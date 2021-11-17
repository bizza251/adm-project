from torch import nn
from constants import ENCODER_D_MODEL, ENCODER_LAYERS


class GraphEncoderTransformer(nn.Module):
    def __init__(self, d_model=ENCODER_D_MODEL, 
            num_layers=ENCODER_LAYERS,
            nhead=8,
            dropout_p=0.2,
            activation=nn.Relu()) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout_p = dropout_p
        self.activation = activation
        encoder_layer = nn.modules.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout_p, batch_first=True, norm_first=True,   
            activation=self.activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.modules.TransformerEncoder(encoder_layer=encoder_layer, num_layers=6, norm=encoder_norm)