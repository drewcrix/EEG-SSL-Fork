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
from torch_geometric.nn import SAGPooling
from torch.nn import Parameter

dataset = PhysionetMI() #working for physionetMT() must change for BIDS
subject = 1
hand_runs = [4, 8, 12]  # Runs corresponding to left and right hand movements
feet_runs = [6, 10, 14]  # Runs corresponding to feet movements

def adjacency_bids(dataset, subject): #set the montage for specific dataset and subject passed through here
    json_path = f"EEG-SSL/data/on/{dataset}/sub-{int(subject):03d}/eeg/sub-{int(subject):03d}_task-eyesclosed_eeg.json"
    
    with open(json_path, "r") as f:
        meta = json.load(f)
    
    scheme = (meta.get("EEGPlacementScheme") or "").lower()
    montage_name = "standard_1020" if "10-20" in scheme or "1020" in scheme else "standard_1005"

    montage = mne.channels.make_standard_montage(montage_name)
    pos = montage.get_positions()["ch_pos"]
    
    coords_2d = np.array([[pos[ch][0], pos[ch][1]] for ch in ch_names if ch in pos], dtype=float)
    kept_names = [ch for ch in ch_names if ch in pos]
    
    A = kneighbors_graph(coords_2d, n_neighbors=min(8, len(kept_names)-1),
                         mode="connectivity", include_self=True).toarray()
    A_matrix = torch.tensor(A, dtype=torch.float32)  
    edge_index, edge_weight = dense_to_sparse(A_matrix)

    return A, A_matrix, edge_index

def _load_subject_raw(data, subj, handrun, feetrun, preload=True):
    filenames = create_paths(subj, handrun, feetrun, data)
    raws = {"hand": [], "feet": []}
    raw_fname_hand = filenames['hand']
    raw_fname_feet = filenames['feet'] 
    for runname in raw_fname_hand:
        raws["hand"].append(raw_info(read_raw_edf(runname, preload=preload, verbose="ERROR")))
    for runname in raw_fname_feet:
        raws["feet"].append(raw_info(read_raw_edf(runname, preload=preload, verbose="ERROR")))
    return raws

def create_paths(subject, hand_runs, feet_runs, dataset):
    base_dir = Path(__file__).parent #directory above the current file
    subject_dir = os.path.join(base_dir, f"data/subject_{subject}") #created path for this subject's data
    path_hands = os.path.join(subject_dir, "hand_runs") #created path for hand runs data
    path_feet = os.path.join(subject_dir, "feet_runs") #created path for feet runs data

    dataset._load_data(subject, feet_runs, path=path_feet) #loads the feet runs locally for this subject
    dataset._load_data(subject, hand_runs, path=path_hands) #loads the hand runs locally for this subject

    #make a list of each downloaded run filename for this subject
    filenames = {
        "hand": sorted([str(p) for p in Path(path_hands).rglob("*.edf")]),
        "feet": sorted([str(p) for p in Path(path_feet).rglob("*.edf")]),
    }

    return filenames

def raw_info(raw):
    raw.rename_channels(lambda x: x.strip("."))
    raw.rename_channels(lambda x: x.upper())
    # fmt: off
    renames = {
        "AFZ": "AFz", "PZ": "Pz", "FPZ": "Fpz", "FCZ": "FCz", "FP1": "Fp1", "CZ": "Cz",
        "OZ": "Oz", "POZ": "POz", "IZ": "Iz", "CPZ": "CPz", "FP2": "Fp2", "FZ": "Fz",
    }
    # fmt: on
    raw.rename_channels(renames)
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"))
    return raw

subject_raws = _load_subject_raw(dataset, subject, hand_runs, feet_runs, preload=True) #loads the raw data for this subject by loading the downloaded .edf files

raw = subject_raws['hand'][0] 
data = raw.get_data() #this has shape (64, 20000) for 64 channels and 2minutes of data at 160Hz sampling rate


def get_adjacency_matrix(raw, n_neighbors=8):
    montage = raw.get_montage()
    positions = montage.get_positions()
    pos_2d = {ch_name: np.array(pos[:2]) for ch_name, pos in positions["ch_pos"].items()}
    ch_names = list(pos_2d.keys())
    coords_2d = np.array([pos_2d[ch] for ch in ch_names])
    A = kneighbors_graph(coords_2d, n_neighbors=n_neighbors, mode='connectivity', include_self=False).toarray()
    
    A_matrix = torch.tensor(A, dtype=torch.float32)  
    edge_index, edge_weight = dense_to_sparse(A_matrix) #getting the edge_index matrix of size (2, num_of_edges) and edge weight vector of size (num_of_edges)
    
    return A, edge_index, edge_weight 


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
