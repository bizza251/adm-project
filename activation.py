from typing import Optional
import torch
from torch import nn
from torch.functional import Tensor
from torch.nn.functional import _in_projection_packed, dropout, linear, softmax
import math


def multi_head_attn(
    query: Tensor,                  # (N, TL, D)
    key: Tensor,                    # (N, SL, D)
    value: Tensor,                  # (N, SL, D)
    in_proj_weight: Tensor,         # (3 * D, D)
    in_proj_bias: Tensor,           # (3 * D, ),
    out_proj_weight: Tensor,        # (HD, D)
    out_proj_bias: Tensor = None,          # (D, )
    nhead: int = 1,
    mask: Tensor = None,                   # (N, TL, SL)
    dropout_p: float = 0.0,
    training: bool = True):

    bsz, tgt_len, embd_dim = query.shape
    _, src_len, _ = key.shape
    assert embd_dim % nhead == 0, "Embedding dimension must be divisible for the number of heads"
    head_dim = embd_dim // nhead
    q, k, v  = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    q = q.contiguous().view(tgt_len, bsz * nhead, head_dim).transpose(0, 1)
    k = k.contiguous().view(src_len, bsz * nhead, head_dim).transpose(0, 1)
    v = v.contiguous().view(src_len, bsz * nhead, head_dim).transpose(0, 1)

    q = q / math.sqrt(head_dim)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if mask is not None:
        attn += mask
    attn = softmax(attn, dim=-1)
    if not training:
        dropout_p = 0.0
    if dropout_p > 0.0:
        attn = dropout(attn, p=dropout_p)
    out = torch.bmm(attn, v)
    out = out.transpose(0, 1).contiguous().view(bsz, tgt_len, embd_dim)
    out = linear(out, out_proj_weight, out_proj_bias)
    return out, attn


class MHA(nn.Module):
    def __init__(self, embd_dim, nhead, dropout_p: float = 0.0) -> None:
        super(MHA, self).__init__()
        assert embd_dim % nhead == 0, "Embedding dimension must be divisible for the number of heads."
        self. embd_dim = embd_dim
        self.nhead = nhead
        self.dropout_p = dropout_p
        self.in_proj_weight = nn.parameter.Parameter(torch.empty((3 * embd_dim, embd_dim)))
        self.in_proj_bias = nn.parameter.Parameter(torch.zeros((3 * embd_dim, )))
        self.out_proj_weight = nn.parameter.Parameter(torch.empty((embd_dim, embd_dim)))
        self.out_proj_bias = nn.parameter.Parameter(torch.zeros((embd_dim, )))

        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj_weight)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, need_weights: bool = True, attn_mask: Optional[Tensor] = None, *args, **kwargs):
        out, attn = multi_head_attn(query, key, value, self.in_proj_weight, self.in_proj_bias,
         self.out_proj_weight, self.out_proj_bias, self.nhead, attn_mask,
            self.dropout_p, self.training) 
        if need_weights:
            return out, attn
        else:
            return out


if __name__ == '__main__':
    bsz, src_len, embd_dim, nhead = 4, 100, 128, 4
    mha = MHA(embd_dim, nhead, dropout_p=0.1)
    src = torch.rand(bsz, src_len, embd_dim)
    memory = torch.rand(bsz, 10, embd_dim)
    out, attn_w = mha(src, memory, memory)
    assert out.shape == src.shape
    