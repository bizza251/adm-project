from typing import Optional
from torch import nn
import torch
from torch.functional import Tensor
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.transformer import TransformerEncoder
from activation import MHA
from constants import DECODER_LAYERS, ENCODER_D_MODEL, ENCODER_LAYERS
import math
import torch.nn.functional as F


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[..., 0::2] = torch.sin(position * div_term)
        pe[..., 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return x + self.pe[:, :x.size(1)]


class TSPTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout_p=0.1, activation=F.relu, layer_norm_eps=0.00001, norm_first=False) -> None:
        super().__init__(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout_p, activation=activation, layer_norm_eps=layer_norm_eps, batch_first=True, norm_first=norm_first)
        self.self_attn = MHA(d_model, nhead, dropout_p)


class TSPTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout_p=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, norm_first=False) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.norm_first = norm_first

        self.self_attn = MHA(d_model, nhead)
        self.multihead_attn = MHA(d_model, nhead)
        self.last_attn = MHA(d_model, 1)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout_p)
        self.dropout2 = Dropout(dropout_p)

    def _self_attn_blk(self, h_t: Tensor):
        out, _ = self.self_attn(h_t, h_t, h_t)
        return self.norm1(h_t + out)

    def _query_attn_blk(self, query: Tensor, memory: Tensor, node_mask: Optional[Tensor] = None):
        out, _ = self.multihead_attn(query, memory, memory, attn_mask=node_mask)
        return self.norm2(query + out)

    def _ff_blk(self, x: Tensor):
        out = self.activation(self.dropout1(self.linear1(x)))
        return self.norm3(out + self.activation(self.dropout2(self.linear2(out))))        

    def forward(self, h_t: Tensor, memory: Tensor, node_mask: Optional[Tensor] = None,):
        query = self._self_attn_blk(h_t)
        out = self._query_attn_blk(query, memory, node_mask)
        return self._ff_blk(out)


class TSPTransformerDecoderLayerFinal(TSPTransformerDecoderLayer):
    def __init__(self, d_model, nhead, c=10) -> None:
        super().__init__(d_model, nhead)
        self.in_proj = nn.Linear(in_features=2 * d_model, out_features=2 * d_model)
        self.c = c

    def forward(self, h_t: Tensor, memory: Tensor, node_mask: Tensor):
        # decoding step 1 -> positional encoding
        # decoding step 2
        query = self._self_attn_blk(h_t)

        # decoding step 3
        out = self._query_attn_blk(query, memory, node_mask)
        
        # decodinig step 4 
        q, k = self.in_proj(torch.cat((out, memory), dim=-1)).chunk(2, dim=-1)
        q = q / math.sqrt(self.d_model)
        attn = torch.bmm(q, k)
        attn += node_mask
        probs = F.softmax(self.c * torch.tanh(attn))
        return probs


class TSPTransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout_p=0.1, layers=2, c=10) -> None:
        super().__init__()
        self.decoder_layers = nn.ModuleList([
            TSPTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout_p) for _ in range(layers - 1)])
        self.final_layer = TSPTransformerDecoderLayerFinal(d_model, nhead, c)
    
    def forward(self, h_t: Tensor, memory: Tensor, node_mask: Tensor):
        output = h_t

        for mod in self.decoder_layers:
            output = mod(output, memory, node_mask=node_mask)

        probs = self.final_layer(output, memory, node_mask)

        return probs


class TSPTransformer(nn.Module):
    def __init__(self,
        in_features=2,
        d_model=128, 
        nhead=4,
        dim_feedforward=1024,
        dropout_p=0.1,
        num_encoder_layers=6,
        num_decoder_layers=2,
        c=10) -> None:

        super().__init__()
        self.pos_enc = PositionalEncoding(d_model, max_len=10000)
        self.input_ff = nn.Linear(in_features=in_features, out_features=d_model)
        encoder_layer = TSPTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout_p)
        enc_norm = LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, enc_norm)
        self.decoder = TSPTransformerDecoder(d_model, nhead, dim_feedforward, dropout_p, num_decoder_layers, c)


    def encode(self, src):
        src = self.input_ff(src)
        memory = self.encoder(src)
        return memory

    def decode(self, memory, visited_node_mask=None):
        bsz, nodes = memory.shape[:2]
        zero_to_bsz = torch.arange(bsz)
        pe = self.pos_enc(torch.zeros(1, nodes, 1)).to(memory.device)

        idx_start_placeholder = torch.randint(low=0, high=nodes, size=(bsz,)).to(memory.device)
        h_start = memory[zero_to_bsz, idx_start_placeholder].view(bsz, 1, -1) + pe[:, 0]
        
        # initialize mask of visited cities
        visited_node_mask = torch.zeros(bsz, nodes + 1, device=memory.device)
        visited_node_mask[zero_to_bsz, idx_start_placeholder] = float("-inf")

        # list that will contain Long tensors of shape (bsz,) that gives the idx of the cities chosen at time t
        tours = []

        # construct tour recursively
        h_t = h_start
        for t in range(nodes):
            
            # compute probability over the next node in the tour
            prob_next_node = self.decoder(h_t, memory, node_mask=visited_node_mask) # size(prob_next_node)=(bsz, nodes+1)
            prob_next_node = nn.functional.softmax(prob_next_node, dim=1)
            
            # choose node with highest probability
            idx = torch.argmax(prob_next_node, dim=1) # size(query)=(bsz,)

            # update embedding of the current visited node
            h_t = memory[zero_to_bsz, idx] # size(h_start)=(bsz, dim_emb)
            h_t = h_t + self.PE[t + 1].expand(bsz, self.dim_emb)
            
            # update tour
            tours.append(idx)

            # update masks with visited nodes
            visited_node_mask = visited_node_mask.clone()
            visited_node_mask[zero_to_bsz, idx] = float("-inf")

        # convert the list of nodes into a tensor of shape (bsz,num_cities)
        tours = torch.stack(tours, dim=1) # size(col_index)=(bsz, nodes)
        
        return tours

    def forward(self, x):
        memory = self.encode(x)
        tour = self.decode(memory)
        return tour
        

if __name__ == '__main__':
    model = TSPTransformer()
    graph = torch.randint(low=int(-1e10), high=int(1e10 + 1), size=(3, 10, 2), dtype=torch.float32)
    out = model(graph)
    pass