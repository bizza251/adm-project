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
from models.layer import CustomPositionalEncoding, CustomSinPositionalEncoding, SinPositionalEncoding, get_positional_encoding
from scipy.optimize import linear_sum_assignment
from models.utility import TourLoss, get_node_mask
from math import sqrt
from torch.distributions import Categorical



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
    _, key_value_len, _ = key.shape
    assert embd_dim % nhead == 0, "Embedding dimension must be divisible for the number of heads"
    head_dim = embd_dim // nhead
    k, v = key, value
    q = query.contiguous().view(tgt_len, bsz * nhead, head_dim).transpose(0, 1)
    k = k.contiguous().view(key_value_len, bsz * nhead, head_dim).transpose(0, 1)
    v = v.contiguous().view(key_value_len, bsz * nhead, head_dim).transpose(0, 1)

    attn = torch.bmm(q, k.transpose(-2, -1))
    attn /= sqrt(embd_dim)
    if mask is not None:
        mask = torch.repeat_interleave(mask, nhead, dim=0)
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
        use_kv_proj: float = True,
        is_self_attn: float = True) -> None:
        
        super().__init__()
        assert embd_dim % nhead == 0, "Embedding dimension must be divisible for the number of heads."
        self.embd_dim = embd_dim
        self.nhead = nhead
        self.dropout_p = dropout_p

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


    def forward(
        self, 
        query: Tensor, 
        key: Tensor, 
        value: Tensor, 
        attn_mask: Optional[Tensor] = None, 
        *args, 
        **kwargs):
        if hasattr(self, 'qkv_proj'):
            qkv = self.qkv_proj(query)
            query, key, value = torch.split(qkv, self.embd_dim, -1)
        else:
            if hasattr(self, 'q_proj'):
                query = self.q_proj(query)
            if kwargs.get('cached_key_value', None):
                key, value = kwargs['cached_key_value']
            elif hasattr(self, 'kv_proj'):
                kv = self.kv_proj(key)
                key, value = torch.split(kv, self.embd_dim, -1)
            
        out, attn = custom_multi_head_attn(query, key, value, self.out_proj_weight, self.out_proj_bias, self.nhead, attn_mask,
            self.dropout_p, self.training) 
        return out, attn, (key, value)



class TransformerFeedforwardBlock(nn.Module):

    def __init__(self, d_model, dim_feedforward, activation=F.relu, layer_norm_eps=1e-5, dropout_p=0.1) -> None:
        super().__init__()
        self.activation = activation
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm = LayerNorm(d_model, layer_norm_eps)
        self.dropout = Dropout(dropout_p)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
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
        use_feedforward_block=True,
        use_q_residual=True,
        is_self_attn=True) -> None:

        super().__init__()

        self.attn = CustomMHA(d_model, nhead, dropout_p, use_q_proj, is_self_attn=is_self_attn)
        self.norm = LayerNorm(d_model, layer_norm_eps)
        if use_feedforward_block:
            self.ff_block = TransformerFeedforwardBlock(
            d_model,
            dim_feedforward,
            activation,
            layer_norm_eps,
            dropout_p
           ) 
        self.use_feedforward_block = use_feedforward_block
        self.use_q_residual = use_q_residual    # True -> vanilla transformer; False -> ours


    def forward(self, query: Tensor, key_value: Tensor, attn_mask: Tensor = None, **kwargs):
        attn_out, attn_weight, cached_key_value = self.attn(query, key_value, key_value, need_weights=True, attn_mask=attn_mask, **kwargs)
        residual = query if self.use_q_residual else key_value
        out = self.norm(residual + attn_out)
        if self.use_feedforward_block:
            out = self.ff_block(out, **kwargs)
        return out, attn_weight, cached_key_value



class TSPCustomEncoderLayer(nn.Module):
    def __init__(self, 
        d_model, 
        nhead, 
        dim_feedforward=1024, 
        activation=F.relu, 
        layer_norm_eps=1e-5,
        norm_first=False,
        add_cross_attn=True,
        use_q_proj_ca=False,
        use_feedforward_block_sa=False,
        use_feedforward_block_ca=True,
        use_q_residual_sa=True,
        use_q_residual_ca=True,
        dropout_p_sa=0.1,
        dropout_p_ca=0.1) -> None:

        super().__init__()
        self.encoder_block = TSPCustomEncoderBlock(
            d_model, 
            nhead, 
            dim_feedforward, 
            dropout_p_sa, 
            activation, 
            layer_norm_eps,
            norm_first,
            True,
            use_feedforward_block_sa,
            use_q_residual=use_q_residual_sa)
    
        self.add_cross_attn = add_cross_attn
        if add_cross_attn:
            self.encoder_block_ca = TSPCustomEncoderBlock(
                d_model, 
                nhead, 
                dim_feedforward, 
                dropout_p_ca, 
                activation, 
                layer_norm_eps,
                norm_first,
                use_q_proj_ca,
                use_feedforward_block_ca,
                use_q_residual=use_q_residual_ca,
                is_self_attn=False)

        self.use_q_residual = use_q_residual_sa and use_q_residual_ca


    def forward(self, query: Tensor, key_value: Tensor, attn_mask: Tensor = None, **kwargs):
        if self.use_q_residual:
            # vanilla transformer: query comes from decoder; key_value from encoder
            query, attn_weight, _ = self.encoder_block(query, query, attn_mask=None, **kwargs)
            attn_out = query
        else:
            # our implementation: query is the pos encoding, key_value is the output of the self-attention
            key_value, attn_weight, _ = self.encoder_block(key_value, key_value, attn_mask=None, **kwargs)
            attn_out = key_value
        if self.add_cross_attn:
            attn_out, attn_weight, cached_key_value = self.encoder_block_ca(query, key_value, attn_mask, **kwargs)
        else:
            cached_key_value = None
        return attn_out, attn_weight, cached_key_value



class TSPCustomEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout_p=0.1, activation=F.relu, layers=2, 
            layer_norm_eps=1e-5, norm_first=False, add_cross_attn=True, use_q_proj_ca=False) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            TSPCustomEncoderLayer(
                d_model, 
                nhead, 
                dim_feedforward, 
                activation, 
                layer_norm_eps, 
                norm_first,
                add_cross_attn,
                use_q_proj_ca,
                use_feedforward_block_sa=False,
                use_feedforward_block_ca=True,
                use_q_residual_sa=False,
                use_q_residual_ca=False,
                dropout_p_sa=dropout_p,
                dropout_p_ca=dropout_p,
                ) for _ in range(layers)])
    

    def forward(self, query: Tensor, key_value: Tensor, attn_mask: Tensor = None):
        output, attn_weight = key_value, None

        for mod in self.layers:
            output, attn_weight, _ = mod(query, output, attn_mask)
        
        return output, attn_weight


class TSPCustomTransformer(nn.Module):

    def _handle_sin_pe(self):
        if type(self.pe) is SinPositionalEncoding:
            # ugly workaround
            from functools import partial
            fwd = partial(self.pe.forward, add_to_input=False)
            self.pe.forward = fwd


    def __init__(self,
        in_features=2,
        d_model=128, 
        nhead=4,
        dim_feedforward=512,
        dropout_p=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        norm_first=False,
        num_hidden_encoder_layers=2,
        sinkhorn_tau=5e-2,
        sinkhorn_i=20,
        add_cross_attn=True,
        use_q_proj_ca=False,
        positional_encoding='sin',
        use_lsa_eval=True,
        **kwargs) -> None:

        super().__init__()
        self.pe = get_positional_encoding(positional_encoding, d_model)
        self.input_ff = nn.Linear(in_features=in_features, out_features=d_model)
        self.input_norm = nn.LayerNorm(d_model, layer_norm_eps)
        self.pos_enc_norm = nn.LayerNorm(d_model, layer_norm_eps)
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
        
        self.head = TSPCustomEncoderLayer(
            d_model,
            1,
            dim_feedforward,
            activation,
            layer_norm_eps,
            norm_first,
            True,
            use_q_proj_ca,
            use_feedforward_block_sa=False,
            use_feedforward_block_ca=False,
            use_q_residual_sa=False,
            use_q_residual_ca=False,
            dropout_p_sa=dropout_p,
            dropout_p_ca=0.)

        self.d_model = d_model
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_k = d_model / nhead
        self.sinkhorn_tau = sinkhorn_tau
        self.sinkhorn_i = sinkhorn_i
        self.use_lsa_eval = use_lsa_eval
        
        self._handle_sin_pe()


    @classmethod
    def from_args(cls, args):
        activation = getattr(F, args.activation)
        return cls(
            in_features=args.in_features,
            d_model=args.d_model, 
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            dropout_p=args.dropout_p,
            activation=activation,
            layer_norm_eps=args.layer_norm_eps,
            norm_first=args.norm_first,
            num_hidden_encoder_layers=args.num_hidden_encoder_layers,
            sinkhorn_tau=args.sinkhorn_tau,
            sinkhorn_i=args.sinkhorn_i,
            add_cross_attn=args.add_cross_attn,
            use_q_proj_ca=args.use_q_proj_ca,
            positional_encoding=args.positional_encoding,
            use_lsa_eval=args.use_lsa_eval)


    def encode(self, key_value, attn_mask=None):
        key_value = self.input_ff(key_value)
        query = self.pe(key_value)
        query = query.expand(len(key_value), *query.shape[1:])
        memory, attn_weight = self.encoder(query, key_value, attn_mask)
        memory, attn_weight, _ = self.head(query, memory, attn_mask)
        return memory, attn_weight


    def forward(self, x, attn_mask=None):
        _, attn_matrix = self.encode(x, attn_mask)
        attn_matrix = sinkhorn(attn_matrix, self.sinkhorn_tau, self.sinkhorn_i)
        bsz, nodes = x.shape[:2]
        tour = torch.empty((bsz, nodes), requires_grad=False)
        if self.training or not self.use_lsa_eval:
            # build tour using soft permutation matrix with sinkhorn algorithm
            node_idx = torch.arange(nodes, device=x.device).expand(bsz, -1)
            idx = torch.argmax(attn_matrix, dim=2)
            tour = torch.gather(node_idx, 1, idx)
        else:
            # build tour using hard permutation matrix with hungarian algorithm
            for i in range(tour.shape[0]):
                tour[i] = torch.tensor(linear_sum_assignment(-attn_matrix[i].detach().cpu().numpy())[1])
        tour = torch.cat((tour, tour[:, 0:1]), dim=1)
        return tour.cpu().to(torch.long), attn_matrix
        


class TSPTransformer(nn.Module):

    @classmethod
    def from_args(cls, args):
        activation = getattr(F, args.activation)
        return cls(
            in_features=args.in_features,
            d_model=args.d_model, 
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            dropout_p=args.dropout_p,
            activation=activation,
            layer_norm_eps=args.layer_norm_eps,
            norm_first=args.norm_first,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_hidden_decoder_layers,
            positional_encoding='sin')


    def __init__(self,
        in_features=2,
        d_model=128, 
        nhead=4,
        dim_feedforward=512,
        dropout_p=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        norm_first=False,
        num_encoder_layers=3,
        num_hidden_decoder_layers=None,
        positional_encoding='sin',
        **kwargs) -> None:

        super().__init__()
        num_decoder_layers = num_hidden_decoder_layers if num_hidden_decoder_layers is not None else num_encoder_layers - 1
        self.pe = get_positional_encoding(positional_encoding, d_model)
        self.input_ff = nn.Linear(in_features=in_features, out_features=d_model)
        self.encoder = nn.ModuleList([
            TSPCustomEncoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                activation,
                layer_norm_eps,
                add_cross_attn=False,
                use_feedforward_block_sa=True,
                dropout_p_sa=dropout_p,
                dropout_p_ca=dropout_p,
            )
            for _ in range(num_encoder_layers)
        ])

        self.decoder = nn.ModuleList([
            TSPCustomEncoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                activation,
                layer_norm_eps,
                add_cross_attn=True,
                use_q_proj_ca=True,
                use_feedforward_block_sa=False,
                use_feedforward_block_ca=True,
                dropout_p_sa=dropout_p,
                dropout_p_ca=dropout_p,
            )
            for _ in range(num_decoder_layers)
        ])   

        self.head = TSPCustomEncoderLayer(
                d_model,
                1,
                None,
                activation,
                layer_norm_eps,
                add_cross_attn=True,
                use_q_proj_ca=True,
                use_feedforward_block_sa=False,
                use_feedforward_block_ca=False,
                dropout_p_sa=dropout_p,
                dropout_p_ca=0.
            )

        self.start_node = nn.Parameter(torch.rand(d_model))
        self.register_buffer('PE', self.pe(torch.zeros(1, 5000, d_model)))    

    
    def encode(self, x):
        for layer in self.encoder:
            x, _, _ = layer(x, x)
        return x


    def decode(self, query, key_value, mask=None, key_value_cache=None):
        key_value_cache = key_value_cache if key_value_cache else []
        use_cache = any(key_value_cache)
        for i, layer in enumerate(self.decoder):
            query, _, cached_key_value = layer(query, key_value, mask, cached_key_value=key_value_cache[i] if use_cache else None)
            key_value_cache.append(cached_key_value)
        last_cached_key_value = key_value_cache[-1] if use_cache else None
        attn_out, attn_weight, cached_key_value = self.head(query, key_value, mask, cached_key_value=last_cached_key_value)
        if not use_cache:
            key_value_cache.append(cached_key_value)
        return attn_out, attn_weight, key_value_cache


    def forward(self, x):
        x = self.input_ff(x)
        key_value = self.encode(x)
        query = self.start_node.expand(len(x), 1, -1)
        bsz, n_nodes, _ = x.shape
        zero2bsz = torch.arange(bsz)
        tour = torch.empty((bsz, n_nodes + 1), dtype=torch.long)
        log_probs = torch.empty((bsz, n_nodes), device=x.device)
        visited_node_mask = torch.zeros((bsz, 1, n_nodes), dtype=x.dtype, device=x.device)
        key_value_cache = None
        for t in range(n_nodes - 1):
            if t > 0:
                query = key_value[zero2bsz, idxs].view(bsz, 1, -1)
            query_pe = query + self.PE[:, t]
            _, attn_weight, key_value_cache = self.decode(query_pe, key_value, visited_node_mask, key_value_cache=key_value_cache)
            if self.training:
                idxs = Categorical(probs=attn_weight).sample()
            else:
                idxs = torch.argmax(attn_weight, dim=-1)
            idxs = idxs.view(-1)
            tour[:, t] = idxs
            log_probs[:, t] = attn_weight[zero2bsz, :, idxs].view(-1)
            visited_node_mask[zero2bsz, :, idxs] += -torch.inf
            if t == n_nodes - 2:
                last_idxs = torch.nonzero(visited_node_mask == 0)[:, -1]
                tour[:, t + 1] = last_idxs
                tour[:, -1] = tour[:, 0]
                log_probs[:, t + 1] = attn_weight[zero2bsz, :, last_idxs].view(-1)
        return tour, log_probs.sum(dim=-1)



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