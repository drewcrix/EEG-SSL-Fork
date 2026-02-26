"""
tests/test_latent_alignment.py

Unit tests for LatentAlignement.py.

LatentProjector aligns GNN embeddings from different electrode configurations
into a shared latent space. Tests cover output shape, gradient flow, and the
multi-layout concat path the module is designed for.
"""

import pytest
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from LatentAlignement import LatentProjector


# ── output shape ─────────────────────────────────────────────────────────────

def test_output_shape_single_tensor():
    proj = LatentProjector(z_dim=64, outputdim=128, hdim=256)
    z    = torch.randn(8, 64)   # (batch, z_dim)
    out  = proj([z])
    assert out.shape == (8, 128)


def test_output_shape_multi_layout_concat():
    # two GNN outputs from different electrode configs, same z_dim
    proj = LatentProjector(z_dim=32, outputdim=64, hdim=128)
    z1   = torch.randn(4, 32)
    z2   = torch.randn(6, 32)
    out  = proj([z1, z2])
    # concat along batch: (10, 32) → project → (10, 64)
    assert out.shape == (10, 64)


def test_output_shape_three_layouts():
    proj = LatentProjector(z_dim=16, outputdim=32, hdim=64)
    tensors = [torch.randn(3, 16) for _ in range(3)]
    out = proj(tensors)
    assert out.shape == (9, 32)


# ── dtype ─────────────────────────────────────────────────────────────────────

def test_output_dtype():
    proj = LatentProjector(z_dim=32, outputdim=64, hdim=64)
    out  = proj([torch.randn(4, 32)])
    assert out.dtype == torch.float32


# ── gradient flow ─────────────────────────────────────────────────────────────

def test_gradient_flows():
    proj = LatentProjector(z_dim=32, outputdim=64, hdim=64)
    z    = torch.randn(4, 32, requires_grad=True)
    out  = proj([z])
    out.sum().backward()
    assert z.grad is not None
    for name, p in proj.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"


def test_parameters_update_on_step():
    proj  = LatentProjector(z_dim=16, outputdim=32, hdim=32)
    opt   = torch.optim.SGD(proj.parameters(), lr=0.01)
    z     = torch.randn(4, 16)
    before = {n: p.clone() for n, p in proj.named_parameters()}
    loss  = proj([z]).sum()
    loss.backward()
    opt.step()
    for name, p in proj.named_parameters():
        assert not torch.equal(p, before[name]), f"{name} didn't update"


# ── architecture ──────────────────────────────────────────────────────────────

def test_projector_has_two_layers():
    proj = LatentProjector(z_dim=16, outputdim=32, hdim=64)
    assert hasattr(proj.net, 'fc1')
    assert hasattr(proj.net, 'fc2')


def test_projector_hidden_dim_respected():
    proj = LatentProjector(z_dim=16, outputdim=32, hdim=64)
    assert proj.net.fc1.out_features == 64
    assert proj.net.fc2.out_features == 32


def test_projector_different_zdim_outputdim():
    for z, o, h in [(8, 512, 256), (256, 8, 128), (64, 64, 64)]:
        proj = LatentProjector(z_dim=z, outputdim=o, hdim=h)
        out  = proj([torch.randn(2, z)])
        assert out.shape == (2, o)


# ── integration with GNN output ───────────────────────────────────────────────

def test_projector_after_gnn_output():
    """
    Simulates the intended use: GNN returns (B, nout, Tp), we global-mean-pool
    to (B, nout) before aligning. Two different montages aligned into one space.
    """
    nout   = 64
    target = 128
    proj   = LatentProjector(z_dim=nout, outputdim=target, hdim=256)

    # two electrode layouts with different Tp (different epoch lengths after CNN)
    layout_a = torch.randn(4, nout, 85).mean(dim=-1)  # (4, nout)
    layout_b = torch.randn(3, nout, 32).mean(dim=-1)  # (3, nout)

    aligned = proj([layout_a, layout_b])
    assert aligned.shape == (7, target)
    assert not torch.isnan(aligned).any()
