import os
import glob
import json

import numpy as np
import pandas as pd
import mne
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from math import ceil
from pathlib import Path

from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import dense_to_sparse

from dn3.trainable.processes import StandardClassification, BaseProcess
from dn3.trainable.models import StrideClassifier, Classifier
from dn3.trainable.layers import Flatten, Permute
from dn3.utils import DN3ConfigException

from GNNencoder import GCNEncoder
from CNNencoder import CNNEncoder

#-----------------------#
#Original GGNStackEncoder
#-----------------------#
"""
class GGNStackEncoder(torch.nn.Module):
    def __init__(self, edge_index, edge_weight=None, hidden_size=512, dropout=None):
        super().__init__()

        self.cnn = CNNEncoder(
            output_channels=max(1, hidden_size // 4),
            kernel_sizes=(128, 64, 32),
            pool_sizes=(5, 3, 2),
            dropout=0.3, #decreased the dropout from 0.5 to 0.3
            stride=1,
            padding="same",
        )
        self.gnn = GCNEncoder(
            nfeat=self.cnn.F,
            nhid=hidden_size,
            nout=hidden_size,
            dropout=dropout,
            pool_ratio=0.5, #updated the pool ratio to 0.5 from 0.9
        )

        self.encoder_h = hidden_size
        self.register_buffer('edge_index', edge_index.cpu())
        self.register_buffer('edge_weight', edge_weight.cpu() if edge_weight is not None else None)


    def forward(self, x):
        h = self.cnn(x)
        _, z_seq = self.gnn(h, self.edge_index, self.edge_weight) #can increase the batch size in the forward pass
        return z_seq

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, strict=True):
        self.load_state_dict(torch.load(path), strict=strict)

    def downsampling_factor(self, samples):
        # CNN pools by (5,3,2) = 30x total; needed by BendingCollegeWav2Vec
        from math import ceil
        for block in (self.cnn.block1, self.cnn.block2, self.cnn.block3):
            p = block.pool.kernel_size
            p = p[0] if isinstance(p, tuple) else p
            samples = ceil(samples / p)
        return samples

    def freeze_features(self, unfreeze=False):
        for param in self.parameters():
            param.requires_grad = unfreeze

    def description(self, sfreq=None, sequence_len=None):
        return f"CNN+GNN Encoder | sfreq={sfreq} | samples={sequence_len} | hidden={self.encoder_h}"
"""
#-----------------------#
#Updated GGNStackEncoder 
#----------------------#
class SingleDatasetEncoder(torch.nn.Module): #EDITS MADE HERE to create a dataset-specific encoder with its own weights and electrode graph, which will be instantiated once per dataset and stored in GGNStackEncoder's ModuleDict. This replaces the previous shared GGNStackEncoder which had one graph and one set of weights for all datasets, which was not ideal since different datasets have different electrode layouts and thus require different graphs.
    """
    CNN+GNN encoder is instantiated once per dataset. Seperate encoder per dataset.
    """

    def __init__(self, edge_index, edge_weight=None, hidden_size=512, dropout=None):
        super().__init__()

        self.cnn = CNNEncoder(
            output_channels=max(1, hidden_size // 4),
            kernel_sizes=(128, 32, 16), #was 128, 64, 32
            pool_sizes=(5, 3, 2),
            dropout=0.3,
            stride=1,
            padding="same",
        )
        self.gnn = GCNEncoder(
            nfeat=self.cnn.F,
            nhid=hidden_size,
            nout=hidden_size,
            dropout=dropout,
            pool_ratio=0.6,
        )

        self.encoder_h = hidden_size
        self.register_buffer('edge_index', edge_index.cpu())
        self.register_buffer('edge_weight', edge_weight.cpu() if edge_weight is not None else None)

    def forward(self, x):
        h = self.cnn(x)
        _, z_seq = self.gnn(h, self.edge_index, self.edge_weight)
        return z_seq

    def downsampling_factor(self, samples):
        from math import ceil
        for block in (self.cnn.block1, self.cnn.block2, self.cnn.block3):
            p = block.pool.kernel_size
            p = p[0] if isinstance(p, tuple) else p
            samples = ceil(samples / p)
        return samples

    def freeze_features(self, unfreeze=False):
        for param in self.parameters():
            param.requires_grad = unfreeze

class GGNStackEncoder(torch.nn.Module): #EDITS MADE HERE to create a stack of dataset-specific encoders instead of one shared encoder with a single graph. Each sub-encoder is a SingleDatasetEncoder with its own weights and electrode graph, and the active one is selected at runtime based on the dataset index provided by NEWBendingCollegeWav2Vec.forward().
    """
    One SingleDatasetEncoder per dataset, each with its own weights and electrode graph,
    stored in an nn.ModuleDict so all are registered as parameters.

    The active encoder is selected at runtime via set_active_dataset(idx) which is called
    by NEWBendingCollegeWav2Vec.forward() before the parent forward pass runs.
    
    """

    def __init__(self, edgeidx_dict, edgeweight_dict, hidden_size=512, dropout=None):
        super().__init__()
        self.encoder_h = hidden_size
        self._active_idx = None  

        # Build one encoder per dataset
        self.encoders = nn.ModuleDict({
            key.replace(' ', '_'): SingleDatasetEncoder(
                edgeidx_dict[key],
                edgeweight_dict.get(key),
                hidden_size,
                dropout,
            )
            for key in edgeidx_dict
        })

    def set_active_dataset(self, idx: int):
        """Called before forward() to select which sub-encoder to use."""
        self._active_idx = idx

    def forward(self, x):
        if self._active_idx is None:
            raise RuntimeError(
                "GGNStackEncoder.set_active_dataset(idx) must be called before forward().")
        key = f"dataset_{self._active_idx}" #make sure the format is correct here!
        if key not in self.encoders:
            raise KeyError(
                f"GGNStackEncoder: no encoder for '{key}'. "
                f"Available: {list(self.encoders.keys())}")
        return self.encoders[key](x)

    def downsampling_factor(self, samples):
        # All sub-encoders share the same CNN architecture so any one gives the right answer
        first = next(iter(self.encoders.values()))
        return first.downsampling_factor(samples)

    def freeze_features(self, unfreeze=False):
        for param in self.parameters():
            param.requires_grad = unfreeze

    def save(self, path):
        """Save all sub-encoder weights in one file."""
        torch.save(self.state_dict(), path)

    def save_encoder(self, dataset_key: str, path: str):
        """Save just one sub-encoder's weights (for test-time loading by dataset)."""
        key = dataset_key.replace(' ', '_')
        torch.save(self.encoders[key].state_dict(), path)

    def load(self, path, strict=True):
        self.load_state_dict(torch.load(path), strict=strict)

    def description(self, sfreq=None, sequence_len=None):
        n = len(self.encoders)
        return (f"GGNStackEncoder | {n} dataset-specific encoders "
                f"| sfreq={sfreq} | samples={sequence_len} | hidden={self.encoder_h}")

def adjacency_bids(top_level, subject=None):
    """
    Build electrode adjacency graph from BIDS sidecar files.
    toplevel: path to the dataset root (e.g. ./on/ds003775).
    """

    json_files = glob.glob(os.path.join(top_level, "**", "*_eeg.json"), recursive=True)
    tsv_files  = glob.glob(os.path.join(top_level, "**", "*_channels.tsv"), recursive=True)

    # filter out derivatives
    json_files = [f for f in json_files if 'derivatives' not in f]
    tsv_files  = [f for f in tsv_files  if 'derivatives' not in f]


    if not json_files:
        raise FileNotFoundError(
            f"adjacency_bids: no JSON sidecar found under {top_level}. "
            f"Check that the dataset is downloaded and the toplevel path is correct. "
            f"Dataset dir searched: {top_level}")

    with open(json_files[0], "r") as f:
        meta = json.load(f)
   
    def montageName(sch):
        sch = sch.lower().replace("-", "").replace(" ", "")
        # map common scheme names to valid MNE montage identifiers
        # standard_1005 is a superset of 10-10 and 10-20 so it works for all three
        _MAP = {
            "1010": "standard_1005",
            "1020": "standard_1020",
            "1005": "standard_1005",
        }
        for key, val in _MAP.items():
            if key in sch:
                return val
        return "standard_1005"  # safe fallback for unknown schemes

    scheme = (meta.get("EEGPlacementScheme") or "").lower()
    montage_name = montageName(scheme)

    montage = mne.channels.make_standard_montage(montage_name)
    pos = montage.get_positions()["ch_pos"]

    df = pd.read_csv(tsv_files[0], sep="\t")
    ch_names = df["name"].dropna().astype(str).tolist()
    
    coords_2d = np.array([[pos[ch][0], pos[ch][1]] for ch in ch_names if ch in pos], dtype=float)
    kept_names = [ch for ch in ch_names if ch in pos]
    
    A = kneighbors_graph(coords_2d, n_neighbors=min(8, len(kept_names)-1),
                         mode="connectivity", include_self=True).toarray()

    A_matrix = torch.tensor(A, dtype=torch.float32)  
    
    edge_index, edge_weight = dense_to_sparse(A_matrix)

    return edge_index, edge_weight
