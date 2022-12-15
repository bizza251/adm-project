from typing import Optional
from torch import nn
import torch
from torch.functional import Tensor
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
import torch.nn.functional as F
from torch.nn.functional import dropout, linear, softmax
from models.activation import sinkhorn
from models.layer import CustomPositionalEncoding, CustomSinPositionalEncoding
from scipy.optimize import linear_sum_assignment
from models.utility import TourLoss, get_node_mask
from math import sqrt



def custom_multi_head_attn(
    query: Tensor,                  # (N, TL, D)
    key: Tensor,                    # (N, SL, D)
    value: Tensor,                  # (N, SL, D)
    out_proj_weight: Tensor = None,        # (HD, D)
    out_proj_bias: Tensor = None,          # (D, )
    nhead: int = 1,
    mask: Tensor = None,                   # (N, TL, SL)
    dropout_p: float = 0.0,
    training: bool = True):

    bsz, tgt_len, embd_dim = query.shape
    _, src_len, _ = key.shape
    assert embd_dim % nhead == 0, "Embedding dimension must be divisible for the number of heads"
    head_dim = embd_dim // nhead
    k, v = key, value
    q = query.contiguous().view(tgt_len, bsz * nhead, head_dim).transpose(0, 1)
    k = k.contiguous().view(src_len, bsz * nhead, head_dim).transpose(0, 1)
    v = v.contiguous().view(src_len, bsz * nhead, head_dim).transpose(0, 1)

    attn = torch.bmm(q, k.transpose(-2, -1))
    attn /= sqrt(embd_dim)
    if mask is not None:
        attn = attn + mask
    attn = softmax(attn, dim=-1)
    if not training:
        dropout_p = 0.0
    if dropout_p > 0.0:
        attn = dropout(attn, p=dropout_p)
    out = torch.bmm(attn, v)
    out = out.transpose(0, 1).contiguous().view(bsz, tgt_len, embd_dim)
    if nhead > 1:
        out = linear(out, out_proj_weight, out_proj_bias)
    return out, attn



class CustomMHA(nn.Module):
    def __init__(self, 
        embd_dim,
        nhead, 
        dropout_p: float = 0.0,
        use_q_proj: float = True, 
        use_kv_proj: float = True) -> None:
        
        super().__init__()
        assert embd_dim % nhead == 0, "Embedding dimension must be divisible for the number of heads."
        self.embd_dim = embd_dim
        self.nhead = nhead
        self.dropout_p = dropout_p

        is_self_attn = use_q_proj and use_kv_proj
        if is_self_attn: 
            self.qkv_proj = nn.Linear(embd_dim, 3 * embd_dim)
        else:
            if use_q_proj: 
                self.q_proj = nn.Linear(embd_dim, embd_dim)
            if use_kv_proj:
                self.kv_proj = nn.Linear(embd_dim, 2 * embd_dim)

        if nhead > 1:
            self.out_proj_weight = nn.parameter.Parameter(torch.empty((embd_dim, embd_dim)))
            self.out_proj_bias = nn.parameter.Parameter(torch.zeros((embd_dim, )))
            nn.init.xavier_uniform_(self.out_proj_weight)
        else:
            self.out_proj_weight = None
            self.out_proj_bias = None


    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_mask: Optional[Tensor] = None, *args, **kwargs):
        if hasattr(self, 'qkv_proj'):
            qkv = self.qkv_proj(query)
            query, key, value = torch.split(qkv, self.embd_dim, -1)
        else:
            if hasattr(self, 'q_proj'):
                query = self.q_proj(query)
            if hasattr(self, 'kv_proj'):
                kv = self.kv_proj(key)
                key, value = torch.split(kv, self.embd_dim, -1)
            
        out, attn = custom_multi_head_attn(query, key, value, self.out_proj_weight, self.out_proj_bias, self.nhead, attn_mask,
            self.dropout_p, self.training) 
        return out, attn



class TransformerFeedforwardBlock(nn.Module):

    def __init__(self, d_model, dim_feedforward, activation=F.relu, layer_norm_eps=1e-5, dropout_p=0.1) -> None:
        super().__init__()
        self.activation = activation
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm = LayerNorm(d_model, layer_norm_eps)
        self.dropout = Dropout(dropout_p)

    def forward(self, x: Tensor) -> Tensor:
        out = self.dropout(self.linear2(self.activation(self.linear1(x))))
        return self.norm(x + out)



class TSPCustomEncoderBlock(nn.Module):
    def __init__(self, 
        d_model, 
        nhead, 
        dim_feedforward=1024, 
        dropout_p=0.1, 
        activation=F.relu, 
        layer_norm_eps=1e-5,
        norm_first=False,
        use_q_proj=True, 
        use_feedforward_block=True) -> None:

        super().__init__()

        self.attn = CustomMHA(d_model, nhead, dropout_p, use_q_proj)
        self.activation = activation
        self.norm = LayerNorm(d_model, layer_norm_eps)
        self.dropout = Dropout(dropout_p)
        if use_feedforward_block:
           self.ff_block = TransformerFeedforwardBlock(
            d_model,
            dim_feedforward,
            activation,
            layer_norm_eps,
            dropout_p
           ) 
        self.use_feedforward_block = use_feedforward_block


    def forward(self, query: Tensor, src: Tensor, attn_mask: Tensor = None):
        attn_out, attn_weight = self.attn(query, src, src, need_weights=True, attn_mask=attn_mask)
        out = self.norm(src + attn_out)
        if self.use_feedforward_block:
            out = self.ff_block(out)
        return out, attn_weight



class TSPCustomEncoderLayer(nn.Module):
    def __init__(self, 
        d_model, 
        nhead, 
        dim_feedforward=1024, 
        dropout_p=0.1, 
        activation=F.relu, 
        layer_norm_eps=1e-5,
        norm_first=False,
        add_cross_attn=True,
        use_q_proj_ca=False,
        use_feedforward_block_ca=True) -> None:

        super().__init__()
        self.encoder_block = TSPCustomEncoderBlock(
            d_model, 
            nhead, 
            dim_feedforward, 
            dropout_p, 
            activation, 
            layer_norm_eps,
            norm_first)
    
        self.add_cross_attn = add_cross_attn
        if add_cross_attn:
            self.encoder_block_ca = TSPCustomEncoderBlock(
                d_model, 
                nhead, 
                dim_feedforward, 
                dropout_p, 
                activation, 
                layer_norm_eps,
                norm_first,
                use_q_proj_ca,
                use_feedforward_block_ca)


    def forward(self, query: Tensor, src: Tensor, attn_mask: Tensor = None):
        attn_out, attn_weight = self.encoder_block(src, src, attn_mask=attn_mask)
        if self.add_cross_attn:
            attn_out, attn_weight = self.encoder_block_ca(query, attn_out, attn_mask)
        return attn_out, attn_weight



class TSPCustomEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout_p=0.1, activation=F.relu, layers=2, 
            layer_norm_eps=1e-5, norm_first=False, add_cross_attn=True, use_q_proj_ca=False) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            TSPCustomEncoderLayer(
                d_model, 
                nhead, 
                dim_feedforward, 
                dropout_p, 
                activation, 
                layer_norm_eps, 
                norm_first,
                add_cross_attn,
                use_q_proj_ca) for _ in range(layers)])
    

    def forward(self, query: Tensor, src: Tensor, attn_mask: Tensor = None):
        output, attn_weight = src, None

        for mod in self.layers:
            output, attn_weight = mod(query, output, attn_mask)
        
        return output, attn_weight


class TSPCustomTransformer(nn.Module):
    def __init__(self,
        in_features=2,
        d_model=128, 
        nhead=4,
        dim_feedforward=1024,
        dropout_p=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        norm_first=False,
        num_hidden_encoder_layers=2,
        sinkhorn_tau=5e-2,
        sinkhorn_i=20,
        add_cross_attn=True,
        use_q_proj_ca=False,
        positional_encoding='custom_sin') -> None:

        super().__init__()
        if positional_encoding == 'custom':
            self.pos_enc = CustomPositionalEncoding(d_model)
        elif positional_encoding == 'custom_sin':
            self.pos_enc = CustomSinPositionalEncoding(d_model)
        self.input_ff = nn.Linear(in_features=in_features, out_features=d_model)
        self.encoder = TSPCustomEncoder(
            d_model, 
            nhead, 
            dim_feedforward, 
            dropout_p, 
            activation, 
            num_hidden_encoder_layers, 
            layer_norm_eps, 
            norm_first,
            add_cross_attn,
            use_q_proj_ca)
        
        self.out_encoder_layer = TSPCustomEncoderLayer(
            d_model,
            1,
            dim_feedforward,
            dropout_p,
            activation,
            layer_norm_eps,
            norm_first,
            True,
            use_q_proj_ca,
            False
        )

        self.d_model = d_model
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_k = d_model / nhead
        self.sinkhorn_tau = sinkhorn_tau
        self.sinkhorn_i = sinkhorn_i


    def encode(self, src, attn_mask=None):
        src = self.input_ff(src)
        query = self.pos_enc(src)
        memory, attn_weight = self.encoder(query, src, attn_mask)
        memory, attn_weight = self.out_encoder_layer(query, memory, attn_mask)
        return memory, attn_weight


    def forward(self, x, attn_mask=None):
        _, attn_matrix = self.encode(x, attn_mask)
        attn_matrix = sinkhorn(attn_matrix, self.sinkhorn_tau, self.sinkhorn_i)
        bsz, nodes = x.shape[:2]
        tour = torch.empty((bsz, nodes))
        if self.training:
            # build tour using soft permutation matrix with sinkhorn algorithm
            node_idx = torch.arange(nodes).expand(bsz, -1)
            idx = torch.argmax(attn_matrix, dim=2)
            tour = torch.gather(node_idx, 1, idx)
        else:
            # build tour using hard permutation matrix with hungarian algorithm
            for i in range(tour.shape[0]):
                tour[i] = torch.tensor(linear_sum_assignment(-attn_matrix[i].detach().numpy())[1])
        tour = torch.cat((tour, tour[:, 0:1]), dim=1)
        return tour.to(torch.int32), attn_matrix
        

if __name__ == '__main__':
    bsz, nodes, dim = 3, 10, 2
    pe = CustomPositionalEncoding(128)
    gt_tours = torch.randint(0, nodes - 1, (bsz, nodes))
    x = pe(gt_tours)
    model = TSPCustomTransformer(nhead=1)
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    graph = torch.randint(low=int(-1e6), high=int(1e6 + 1), size=(bsz, nodes, dim), dtype=torch.float32)
    gt_tour = torch.randint(0, nodes - 1, (bsz, nodes))
    mask = get_node_mask(nodes, torch.tensor(([0, 4], [5, 5], [9, 1])))
    out, attn_matrix = model(graph, mask)
    loss = TourLoss()
    l = loss(attn_matrix, gt_tour)
    l.backward()
    optimizer.step()
    print(out, l)
    assert l.grad_fn is not None