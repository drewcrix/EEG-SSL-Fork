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
A = torch.tensor(adj_matrix, dtype=torch.float32)  
edge_index, edge_weight = dense_to_sparse(A) 


#---------------------------
#Now normalize the adjacency matrix
def normalize_adj(A: np.ndarray, add_self=True) -> torch.Tensor:
    if add_self: #each node connected to itself
        A = A + np.eye(A.shape[0]) # A + AI
    D = np.sum(A, axis=1) #row sum degree vector 
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D)) # diagonal matrix turns the vector D into a diagonal matrix 
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt #performs matrix multiplication. Can also us np.linalg.multi_dot for multiple matrices
    return torch.from_numpy(A_norm).float()

A_norm = normalize_adj(adj_matrix, add_self=True)  # (S,S), #normalization of adjacency matrix actually gets conducted in the GCNConv layer so this is just for visualization here 

print("Normalized adjacency matrix A_norm:")



"""
class GraphConvolution(nn.Module):
 
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
 

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
"""   

class SAGPool(torch.nn.Module):
    def __init__(self,in_channels,ratio=0.8,Conv=GCNConv,non_linearity=torch.tanh):
        super(SAGPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels,1)
        self.non_linearity = non_linearity
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        #x = x.unsqueeze(-1) if x.dim() == 1 else x
        score = self.score_layer(x,edge_index).squeeze()

        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm


class GCNEncoder(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout=0.5, pool_ratio=0.8):
        super().__init__()

        # I think DN3 needs a format for self.encoder_h = nout

        self.gc1 = GCNConv(nfeat, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.rel1 = nn.PReLU()
        self.pool1 = SAGPool(nhid, ratio=pool_ratio, Conv=GCNConv)

        self.gc2 = GCNConv(nhid, nout)
        self.bn2 = nn.BatchNorm1d(nout)
        self.rel2 = nn.PReLU()
        self.pool2 = SAGPool(nout, ratio=pool_ratio, Conv=GCNConv)

        self.drop = nn.Dropout(p=dropout)

    def forward(self, x, edge_index, edge_weight=None, batch=None): #maybe issues with the batch that I need to fix
        # ----- Block 1 -----
        x = self.gc1(x, edge_index, edge_weight=edge_weight)
        x = self.bn1(x)
        x = self.rel1(x)
        x, edge_index, edge_weight, batch = self.pool1(x, edge_index, edge_attr=edge_weight, batch=batch)
        x = self.drop(x)

        # ----- Block 2 -----
        x = self.gc2(x, edge_index, edge_weight=edge_weight)
        x = self.bn2(x)
        x = self.rel2(x)
        x, edge_index, edge_weight, batch = self.pool2(x, edge_index, edge_attr=edge_weight, batch=batch) #sag pooling
        x = self.drop(x)

        # DN3 needs (batch, features, timepoints); global mean pooling here removes the temporal dimension and collapses to a single vector (does CNN fix this?)
        z = global_mean_pool(x, batch)  # global mean pooling

        return x, z

