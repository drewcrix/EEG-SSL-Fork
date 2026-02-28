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

class GGNStackEncoder(torch.nn.Module):
    def __init__(self, edge_index, edge_weight=None, hidden_size=512, dropout=None):
        super().__init__()

        self.cnn = CNNEncoder(
            output_channels=max(1, hidden_size // 4),
            kernel_sizes=(128, 64, 32),
            pool_sizes=(5, 3, 2),
            dropout=0.5,
            stride=1,
            padding="same",
        )
        self.gnn = GCNEncoder(
            nfeat=self.cnn.F,
            nhid=hidden_size,
            nout=hidden_size,
            dropout=dropout,
            pool_ratio=0.9,
        )

        self.encoder_h = hidden_size
        self.register_buffer('edge_index', edge_index.cpu())
        self.register_buffer('edge_weight', edge_weight.cpu() if edge_weight is not None else None)


    def forward(self, x):
        h = self.cnn(x)
        _, z_seq = self.gnn(h, self.edge_index, self.edge_weight)
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