import os
import json
import math
import time
import warnings

import mne
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple
from sklearn.neighbors import kneighbors_graph

warnings.filterwarnings("ignore")

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import SAGPooling
from torch.nn import Parameter

class GCNEncoder(nn.Module):
    """
    GNN encoder for CNN output shaped (B, C, F, Tp).

    - B  : trials (mini-batch size)
    - C  : electrodes (nodes per graph)
    - F  : Feature dim per electrode provided from CNN
    - Tp : Time dimension containing feature strengths over time
"""
    def __init__(self, nfeat: int, nhid: int, nout: int, dropout: float = 0.5, pool_ratio: float = 0.9):
        super().__init__()


        self.gc1 = GCNConv(nfeat, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.rel1 = nn.PReLU()
        self.pool1 = SAGPooling(nhid, ratio=pool_ratio, GNN=GCNConv) #check the batching here. With custom its Conv not GNN

        self.gc2 = GCNConv(nhid, nout)
        self.bn2 = nn.BatchNorm1d(nout)
        self.rel2 = nn.PReLU()

        self.drop = nn.Dropout(p=dropout)
    
    def _repeat_fixed_graph(self, edge_index, edge_weight, num_graphs, C):
        if num_graphs == 1:
            return edge_index, edge_weight

        device = edge_index.device
        E = edge_index.size(1)

        edge_index_big = edge_index.repeat(1, num_graphs)
        offsets = (torch.arange(num_graphs, device=device) * C).repeat_interleave(E)
        edge_index_big = edge_index_big + offsets.unsqueeze(0)

        edge_weight_big = edge_weight.repeat(num_graphs)
        return edge_index_big, edge_weight_big

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor):

        B, C, F, Tp = h.shape
        num_graphs = B * Tp

        x = h.permute(0, 3, 1, 2).contiguous().view(num_graphs * C, F)

        batch = torch.arange(num_graphs, device=h.device, dtype=torch.long).repeat_interleave(C)

        edge_index_big, edge_weight_big = self._repeat_fixed_graph(edge_index, edge_weight, num_graphs, C)

        # ----- Block 1 -----
        x = self.gc1(x, edge_index_big, edge_weight=edge_weight_big)
        x = self.bn1(x)
        x = self.rel1(x)
        x, edge_index_big, edge_weight_big, batch, perm, score = self.pool1(
            x, edge_index_big, edge_attr=edge_weight_big, batch=batch
        ) #check the batching here
        x = self.drop(x)

        # ----- Block 2 -----
        x = self.gc2(x, edge_index_big, edge_weight=edge_weight_big)
        x = self.bn2(x)
        x = self.rel2(x)
        
        x = self.drop(x)

       
        z = global_mean_pool(x, batch)
        features = z.size(-1)   
        z_seq = z.view(B, Tp, features) 
        z_seq = z_seq.permute(0, 2, 1).contiguous() #permute to (B, features, Tp) for consistency with Bendr       

        return x, z_seq


if __name__ == "__main__":
    print("GCNEncoder loaded OK")
