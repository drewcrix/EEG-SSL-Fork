import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import torch
import time 
from moabb.datasets.physionet_mi import PhysionetMI
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
start = time.time()
warnings.filterwarnings("ignore")  # Ignore warnings for cleaner output
import math

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch.nn import Parameter

# ---------------------------
subject = 1
run = 4

dataset = PhysionetMI()

#data = dataset._get_single_subject_data(subject)

#runTest = data["0"]["0"] #the structure is a nested dictionary


raw = dataset._load_one_run(subject, run)


montage = raw.get_montage()

positions = montage.get_positions()


pos_2d = {}

for ch_name in positions["ch_pos"]:
    pos = positions["ch_pos"][ch_name]
    pos_2d[ch_name] = np.array(pos[:2])  # Take only x and y coordinates

#---------------------------
#Construct adjacency matrix from 2D positions using k-NN graph
ch_names = list(pos_2d.keys())
coords_2d = np.array([pos_2d[ch] for ch in ch_names])
A = kneighbors_graph(coords_2d, n_neighbors=8, mode='connectivity', include_self=False).toarray()


def get_global_adjacency_matrix(A, name11,name12,name21,name22,name31,name32): #accounts for global connections
    for i in range(len(ch_names)):
        if ch_names[i]==name11:
            idx11=i
        if ch_names[i]==name12:
            idx12=i
        if ch_names[i]==name21:
            idx21=i
        if ch_names[i]==name22:
            idx22=i
        if ch_names[i]==name31:
            idx31=i
        if ch_names[i]==name32:
            idx32=i
    A[idx11,idx12]=1
    A[idx12,idx11]=1
    A[idx21,idx22]=1
    A[idx22,idx21]=1
    A[idx31,idx32]=1
    A[idx32,idx31]=1
    return A
    

adj_matrix = get_global_adjacency_matrix(A, "FC3", "FC4", "C3", "C4", "CP3", "CP4") #example of adding global connections between homologous regions
print("Adjacency matrix shape:", adj_matrix.shape)
print("Connections for FC5:", adj_matrix[ch_names.index("FC5")]) #prints the connections for FC5 electrode which is the first channel. Row 1 of the adjacency matrix with each corresponding column. Therefore there are 64 electrodes and 64 values in this row. A value of 1 indicates a connection to that electrode, while a value of 0 indicates no connection.


# Convert dense adjacency matrix to edge_index and edge_weight format for PyTorch Geometric
A_matrix = torch.tensor(adj_matrix, dtype=torch.float32)  
edge_index, edge_weight = dense_to_sparse(A_matrix) 


class SAGPool(nn.Module):
    def __init__(self, in_channels, ratio=0.8, Conv=GCNConv, non_linearity=torch.tanh):
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels, 1)
        self.non_linearity = non_linearity

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        score = self.score_layer(x, edge_index).squeeze(-1)  # (N,)

        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]

        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0))
        return x, edge_index, edge_attr, batch


class GCNEncoder(nn.Module):
    """
    GNN encoder for CNN output shaped (B, C, F, Tp).

    - B  : trials (mini-batch size)
    - C  : electrodes (nodes per graph)
    - F  : Feature dim per electrode provided from CNN
    - Tp : Time dimension containing feature strengths over time

    Treat each time slice as a separate graph by folding time into the batch dimension:
      num_graphs = B * Tp, electrodes/nodes per graph = C
    """
    def __init__(self, nfeat: int, nhid: int, nout: int, dropout: float = 0.5, pool_ratio: float = 0.8):
        super().__init__()

        self.gc1 = GCNConv(nfeat, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.rel1 = nn.PReLU()
        self.pool1 = SAGPool(nhid, ratio=pool_ratio, Conv=GCNConv)

        self.gc2 = GCNConv(nhid, nout)
        self.bn2 = nn.BatchNorm1d(nout)
        self.rel2 = nn.PReLU()
        self.pool2 = SAGPool(nout, ratio=pool_ratio, Conv=GCNConv)

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

        # Fold time into batch: (B, C, F, Tp) -> (B, Tp, C, F) -> (B*Tp*C, F)
        # PyG expects (n_out, F) therefore collapsing time and electrodes into channels
        x = h.permute(0, 3, 1, 2).contiguous().view(num_graphs * C, F)

        batch = torch.arange(num_graphs, device=h.device, dtype=torch.long).repeat_interleave(C)

        # disjoint-union adjacency for all graphs in this forward
        edge_index_big, edge_weight_big = self._repeat_fixed_graph(edge_index, edge_weight, num_graphs, C)

        # ----- Block 1 -----
        x = self.gc1(x, edge_index_big, edge_weight=edge_weight_big)
        x = self.bn1(x)
        x = self.rel1(x)
        x, edge_index_big, edge_weight_big, batch = self.pool1(
            x, edge_index_big, edge_attr=edge_weight_big, batch=batch
        )
        x = self.drop(x)

        # ----- Block 2 -----
        x = self.gc2(x, edge_index_big, edge_weight=edge_weight_big)
        x = self.bn2(x)
        x = self.rel2(x)
        x, edge_index_big, edge_weight_big, batch = self.pool2(
            x, edge_index_big, edge_attr=edge_weight_big, batch=batch
        )
        x = self.drop(x)

       
        z = global_mean_pool(x, batch)   
        z_seq = z.view(B, Tp, -1)        

        return x, z_seq
