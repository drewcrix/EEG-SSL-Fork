"""
tests/test_encoders.py

Unit tests for CNNencoder.py and GNNencoder.py.

Checks output shapes, dtype, gradient flow, edge cases, and the
CNN → GNN pipeline that pretrain.py wires together.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from CNNencoder import CNNEncoder, Block
from GNNencoder import GCNEncoder
from torch_geometric.utils import dense_to_sparse
from sklearn.neighbors import kneighbors_graph


# ── helpers ───────────────────────────────────────────────────────────────────

def make_simple_graph(n_nodes=8, n_neighbors=3):
    """Small fully-connected-ish graph for testing the GNN."""
    coords = np.random.rand(n_nodes, 2).astype(np.float32)
    A = kneighbors_graph(coords, n_neighbors=min(n_neighbors, n_nodes - 1),
                         mode="connectivity", include_self=True).toarray()
    ei, ew = dense_to_sparse(torch.tensor(A, dtype=torch.float32))
    return ei, ew


# ── Block ─────────────────────────────────────────────────────────────────────

def test_block_output_shape():
    b = Block(in_ch=1, out_ch=4, kernel_size=9, pool_size=2, dropout=0.0)
    x = torch.randn(16, 1, 128)
    y = b(x)
    # conv(padding=same) preserves T, pool divides by 2
    assert y.shape == (16, 4, 64)


def test_block_output_dtype():
    b = Block(in_ch=1, out_ch=4, kernel_size=9, pool_size=2, dropout=0.0)
    x = torch.randn(4, 1, 64)
    assert b(x).dtype == torch.float32


# ── CNNEncoder ────────────────────────────────────────────────────────────────

def test_cnn_output_shape_default():
    # default: output_channels=4, pools (5,3,2) → Tp = 2560/5/3/2 = 85
    model = CNNEncoder()
    x = torch.randn(2, 64, 2560)
    out = model(x)
    assert out.shape[0] == 2,  "batch dim"
    assert out.shape[1] == 64, "channel dim preserved"
    assert out.shape[2] == model.F, "feature dim = o3"
    assert out.shape[3] == 2560 // 5 // 3 // 2  # Tp = 85


def test_cnn_output_shape_small():
    # use small kernels so we don't need 2560 samples
    model = CNNEncoder(output_channels=2, kernel_sizes=(3, 3, 3),
                       pool_sizes=(2, 2, 2), padding=1)
    x = torch.randn(4, 32, 256)
    out = model(x)
    assert out.shape[0] == 4
    assert out.shape[1] == 32
    assert out.shape[2] == model.F  # 2*2*2=8
    assert out.shape[3] == 256 // 2 // 2 // 2  # 32


def test_cnn_batch_independence():
    # running single and double batch should give the same result for batch[0]
    model = CNNEncoder(output_channels=2, kernel_sizes=(3, 3, 3),
                       pool_sizes=(2, 2, 2), padding=1)
    model.eval()
    x = torch.randn(2, 8, 128)
    with torch.no_grad():
        out_pair   = model(x)
        out_single = model(x[:1])
    assert torch.allclose(out_pair[:1], out_single, atol=1e-5)


def test_cnn_gradient_flows():
    model = CNNEncoder(output_channels=2, kernel_sizes=(3, 3, 3),
                       pool_sizes=(2, 2, 2), padding=1)
    x = torch.randn(2, 8, 128)
    out = model(x)
    out.sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"


def test_cnn_raises_on_wrong_kernel_count():
    with pytest.raises(ValueError):
        CNNEncoder(kernel_sizes=(3, 3))  # must be length 3


def test_cnn_raises_on_mismatched_pool_count():
    with pytest.raises(ValueError):
        CNNEncoder(kernel_sizes=(3, 3, 3), pool_sizes=(2, 2))  # must match


def test_cnn_feature_dim_scales_with_output_channels():
    m8  = CNNEncoder(output_channels=8)
    m16 = CNNEncoder(output_channels=16)
    assert m16.F == 2 * m8.F


# ── GCNEncoder ────────────────────────────────────────────────────────────────

@pytest.fixture
def small_graph():
    return make_simple_graph(n_nodes=8, n_neighbors=3)


def test_gcn_output_shape(small_graph):
    ei, ew = small_graph
    B, C, F, Tp = 2, 8, 16, 10
    model = GCNEncoder(nfeat=F, nhid=32, nout=64, dropout=0.0)
    h = torch.randn(B, C, F, Tp)
    _, z_seq = model(h, ei, ew)
    # z_seq should be (B, nout, Tp)
    assert z_seq.shape == (B, 64, Tp), f"got {z_seq.shape}"


def test_gcn_batch_dim_preserved(small_graph):
    ei, ew = small_graph
    model = GCNEncoder(nfeat=8, nhid=16, nout=32, dropout=0.0)
    for B in [1, 2, 4]:
        h = torch.randn(B, 8, 8, 5)
        _, z = model(h, ei, ew)
        assert z.shape[0] == B


def test_gcn_gradient_flows(small_graph):
    ei, ew = small_graph
    model = GCNEncoder(nfeat=8, nhid=16, nout=32, dropout=0.0)
    h = torch.randn(2, 8, 8, 5)
    _, z = model(h, ei, ew)
    z.sum().backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} has no gradient"


def test_gcn_repeat_fixed_graph():
    # _repeat_fixed_graph should tile edge_index correctly for num_graphs > 1
    model = GCNEncoder(nfeat=4, nhid=8, nout=8)
    C = 4
    # simple self-loop graph on C=4 nodes: 4 edges
    ei = torch.stack([torch.arange(C), torch.arange(C)])  # (2, 4)
    ew = torch.ones(C)

    ei2, ew2 = model._repeat_fixed_graph(ei, ew, num_graphs=3, C=C)
    assert ei2.shape[1] == 3 * C   # 3x edges
    assert ew2.shape[0] == 3 * C
    # second graph offset by C, third by 2*C
    assert ei2[0, C] == C          # first edge of second graph starts at node C


def test_gcn_single_timestep(small_graph):
    # Tp=1 is a degenerate edge case — should still work
    ei, ew = small_graph
    model = GCNEncoder(nfeat=8, nhid=16, nout=32, dropout=0.0)
    h = torch.randn(2, 8, 8, 1)
    _, z = model(h, ei, ew)
    assert z.shape == (2, 32, 1)


# ── CNN → GNN pipeline (end-to-end shape) ────────────────────────────────────

def test_cnn_to_gnn_pipeline():
    """
    Full shape check for the pipeline pretrain.py runs with --use-gnn.
    Input  (B, C, T) -> CNN -> (B, C, F, Tp) -> GNN -> (B, nout, Tp)
    """
    B, C, T = 2, 8, 256
    cnn = CNNEncoder(output_channels=2, kernel_sizes=(3, 3, 3),
                     pool_sizes=(2, 2, 2), padding=1)
    ei, ew = make_simple_graph(n_nodes=C, n_neighbors=3)
    gnn = GCNEncoder(nfeat=cnn.F, nhid=32, nout=64, dropout=0.0)

    x    = torch.randn(B, C, T)
    h    = cnn(x)                     # (B, C, F, Tp)
    _, z = gnn(h, ei, ew)             # (B, 64, Tp)

    assert z.shape[0] == B
    assert z.shape[1] == 64
    assert z.shape[2] == h.shape[3]   # Tp matches

    # permute to (B, Tp, 64) then back to (B, 64, Tp) — just the final step pretrain does
    z_for_context = z  # already (B, encoder_h, Tp)
    assert z_for_context.shape == (B, 64, h.shape[3])
