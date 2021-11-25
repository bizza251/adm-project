from torch import nn
import torch
from torch._C import dtype
from torch.functional import Tensor
from constants import DECODER_LAYERS, ENCODER_D_MODEL, ENCODER_LAYERS
import math


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


class GraphTransformer(nn.Transformer):
    def __init__(self,
        in_features=2,
        d_model=ENCODER_D_MODEL, 
        nhead=8,
        num_encoder_layers=ENCODER_LAYERS,
        num_decoder_layers=DECODER_LAYERS,
        dim_feedforward=1024) -> None:

        super().__init__(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=0.0, batch_first=True)

        self.decoder.norm = None
        self.pos_enc = PositionalEncoding(d_model, max_len=10000)
        self.input_ff = nn.Linear(in_features=in_features, out_features=d_model)


    def encode(self, src):
        src = self.input_ff(src)
        src = self.pos_enc(src)
        memory = self.encoder(src)
        return memory

    def decode(self, memory, visited_node_mask=None):
        bsz, nodes = memory.shape[:2]
        zero_to_bsz = torch.arange(bsz)
        pe = self.pos_enc(torch.zeros(1, nodes, 1)).to(memory.device)

        idx_start_placeholder = torch.randint(low=0, high=nodes, size=(bsz,)).to(memory.device)
        h_start = memory[zero_to_bsz, idx_start_placeholder].view(bsz, 1, -1) + pe[:, 0]
        
        # initialize mask of visited cities
        mask_visited_nodes = torch.zeros(bsz, nodes + 1, device=memory.device).bool() # False
        mask_visited_nodes[zero_to_bsz, idx_start_placeholder] = True

        # list that will contain Long tensors of shape (bsz,) that gives the idx of the cities chosen at time t
        tours = []

        # construct tour recursively
        h_t = h_start
        for t in range(nodes):
            
            # compute probability over the next node in the tour
            prob_next_node = self.decoder(h_t, memory, memory_mask=mask_visited_nodes) # size(prob_next_node)=(bsz, nodes+1)
            prob_next_node = nn.functional.softmax(prob_next_node, dim=1)
            
            # choose node with highest probability
            idx = torch.argmax(prob_next_node, dim=1) # size(query)=(bsz,)

            # update embedding of the current visited node
            h_t = memory[zero_to_bsz, idx] # size(h_start)=(bsz, dim_emb)
            h_t = h_t + self.PE[t + 1].expand(bsz, self.dim_emb)
            
            # update tour
            tours.append(idx)

            # update masks with visited nodes
            mask_visited_nodes = mask_visited_nodes.clone()
            mask_visited_nodes[zero_to_bsz, idx] = True

        # convert the list of nodes into a tensor of shape (bsz,num_cities)
        tours = torch.stack(tours, dim=1) # size(col_index)=(bsz, nodes)
        
        return tours

    def forward(self, x):
        memory = self.encode(x)
        tour = self.decode(memory)
        return tour
        

if __name__ == '__main__':
    model = GraphTransformer()
    graph = torch.randint(low=int(-1e10), high=int(1e10 + 1), size=(3, 10, 2), dtype=torch.float32)
    out = model(graph)
    pass