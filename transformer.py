from typing import Optional
from torch import nn
import torch
from torch.functional import Tensor
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.transformer import TransformerEncoder
from activation import MHA, multi_head_attn
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


class TSPTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout_p=0.1, activation=F.relu, layer_norm_eps=1e-5, norm_first=False) -> None:
        super().__init__()

        self.self_attn = MHA(d_model, nhead, 0.0)
        self.activation = activation
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model, layer_norm_eps)
        self.norm2 = LayerNorm(d_model, layer_norm_eps)
        self.dropout = Dropout(dropout_p)

    def forward(self, h_t: Tensor):
        attn_out, attn_weight = self.self_attn(h_t, h_t, h_t, need_weights=True)
        out = self.norm1(h_t + attn_out)
        out_ff = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = self.norm2(out + out_ff)
        return out, attn_weight


class TSPTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout_p=0.1, activation=F.relu, layers=2, 
            layer_norm_eps=1e-5, norm_first=False) -> None:
        super().__init__()
        self.decoder_layers = nn.ModuleList([
            TSPTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout_p, activation, layer_norm_eps, norm_first) for _ in range(layers)])
    
    def forward(self, h_t: Tensor):
        output, attn_weight = h_t, None

        for mod in self.decoder_layers:
            output, attn_weight = mod(output)

        return output, attn_weight


class TSPTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout_p=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, norm_first=False) -> None:
        super().__init__()

        self.activation = activation
        self.norm_first = norm_first

        self.self_attn = MHA(d_model, nhead)
        self.multihead_attn = MHA(d_model, nhead)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = Dropout(dropout_p)

    def _self_attn_blk(self, h_t: Tensor):
        out, _ = self.self_attn(h_t, h_t, h_t)
        return self.norm1(h_t + out)

    def _query_attn_blk(self, query: Tensor, memory: Tensor, node_mask: Optional[Tensor] = None):
        out, _ = self.multihead_attn(query, memory, memory, attn_mask=node_mask)
        return self.norm2(query + out)

    def _ff_blk(self, x: Tensor):
        out = self.linear2(self.activation(self.dropout(self.linear1(x))))
        return self.norm3(x + out)        

    def forward(self, h_t: Tensor, memory: Tensor, node_mask: Optional[Tensor] = None,):
        query = self._self_attn_blk(h_t)
        out = self._query_attn_blk(query, memory, node_mask)
        return self._ff_blk(out)


class TSPTransformerDecoderLayerFinal(TSPTransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout_p=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, norm_first=False, c=10):
        super().__init__(d_model, nhead, dim_feedforward, dropout_p, activation, layer_norm_eps, norm_first)
        self.in_proj_q = nn.Linear(in_features=d_model, out_features=d_model)
        self.in_proj_k = nn.Linear(in_features=d_model, out_features=d_model)
        self.c = c
        self.nhead = nhead
        self.d_model = d_model

    def forward(self, h_t: Tensor, memory: Tensor, node_mask: Tensor):
        # decoding step 1 -> positional encoding
        # decoding step 2
        query = self._self_attn_blk(h_t)

        # decoding step 3
        h_q = self._query_attn_blk(query, memory, node_mask)
        
        # decodinig step 4 
        q, k = self.in_proj_q(h_q), self.in_proj_k(memory)
        bsz, tgt_len, embd_dim = query.shape
        _, src_len, _ = memory.shape
        head_dim = embd_dim // self.nhead
        q = q.contiguous().view(tgt_len, bsz * self.nhead, head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.nhead, head_dim).transpose(0, 1)
        q = q / math.sqrt(self.d_model)
        attn = torch.bmm(q, k.transpose(-2, -1))
        if self.nhead > 1:
            node_mask = torch.repeat_interleave(node_mask, repeats=self.nhead, dim=0)
        node_mask = node_mask.unsqueeze(1)
        attn = attn + node_mask
        probs = F.softmax(self.c * torch.tanh(attn), dim=-1)
        probs = probs.transpose(0, 1).contiguous().view(bsz, self.nhead, src_len).mean(1)
        return probs


class TSPTransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout_p=0.1, activation=F.relu, layer_norm_eps=1e-5, norm_first=False,
            layers=2, c=10) -> None:
        super().__init__()
        self.decoder_layers = nn.ModuleList([
            TSPTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout_p, 
                activation, layer_norm_eps, norm_first) for _ in range(layers - 1)])
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
        activation=F.relu,
        layer_norm_eps=1e-5,
        norm_first=False,
        num_encoder_layers=6,
        num_decoder_layers=2,
        c=10) -> None:

        super().__init__()
        self.pos_enc = PositionalEncoding(d_model, max_len=10000)
        self.input_ff = nn.Linear(in_features=in_features, out_features=d_model)
        self.encoder = TSPTransformerEncoder(d_model, nhead, dim_feedforward, dropout_p, activation, num_encoder_layers, layer_norm_eps, norm_first)
        self.decoder = TSPTransformerDecoder(d_model, nhead, dim_feedforward, dropout_p, activation, layer_norm_eps, norm_first, num_decoder_layers, c)
        self.d_model = d_model
        self.z = nn.parameter.Parameter(torch.empty(d_model, 1))
        nn.init.xavier_uniform_(self.z)

    def encode(self, src):
        src = self.input_ff(src)
        memory, attn_weight = self.encoder(src)
        return memory, attn_weight

    def decode(self, memory, visited_node_mask=None):
        bsz, nodes = memory.shape[:2]
        zero_to_bsz = torch.arange(bsz)
        pe = self.pos_enc(torch.zeros(1, nodes + 1, 1)).to(memory.device)

        # idx_start_placeholder = torch.randint(low=0, high=nodes, size=(bsz,)).to(memory.device)
        z = self.z.expand(bsz, -1, 1)
        h_start_logits = torch.bmm(memory, z).squeeze()
        h_start_probs = torch.softmax(h_start_logits, dim=-1)
        idx_start_placeholder = torch.argmax(h_start_probs, dim=-1)
        h_start = memory[zero_to_bsz, idx_start_placeholder].view(bsz, 1, -1) + pe[:, 0]
        
        # initialize mask of visited cities
        visited_node_mask = torch.zeros(bsz, nodes, device=memory.device)
        visited_node_mask[zero_to_bsz, idx_start_placeholder] = float("-inf")

        # list that will contain Long tensors of shape (bsz,) that gives the idx of the cities chosen at time t
        tours = [idx_start_placeholder]

        # construct tour recursively
        h_t = h_start
        for t in range(nodes - 2):
            
            # compute probability over the next node in the tour
            prob_next_node = self.decoder(h_t, memory, node_mask=visited_node_mask) # size(prob_next_node)=(bsz, nodes+1)
            prob_next_node = nn.functional.softmax(prob_next_node, dim=1)
            
            # choose node with highest probability
            idx = torch.argmax(prob_next_node, dim=1) # size(query)=(bsz,)

            # update embedding of the current visited node
            h_t = memory[zero_to_bsz, idx] # size(h_start)=(bsz, dim_emb)
            h_t = h_t + pe[:, t + 1].expand(bsz, self.d_model)
            h_t = h_t.unsqueeze(1)
            
            # update tour
            tours.append(idx)

            # update masks with visited nodes
            visited_node_mask = visited_node_mask.clone()
            visited_node_mask[zero_to_bsz, idx] = float("-inf")
        
        last_idx = torch.nonzero(visited_node_mask != float('-inf'))
        tours.append(last_idx[:, 1])

        # convert the list of nodes into a tensor of shape (bsz,num_cities)
        tours = torch.stack(tours, dim=1) # size(col_index)=(bsz, nodes)

        # close the tour
        tours = torch.cat((tours, tours[:, 0:1]), dim=-1)
        
        return tours

    def forward(self, x):
        memory, _ = self.encode(x)
        tour = self.decode(memory)
        return tour
        

if __name__ == '__main__':
    model = TSPTransformer()
    graph = torch.randint(low=int(-1e6), high=int(1e6 + 1), size=(3, 10, 2), dtype=torch.float32)
    out = model(graph)
    print(out)