"""
tests/test_dn3_ext.py

Unit tests for dn3_ext.py:
  - ConvEncoderBENDR: output shape, downsampling factor, gradient
  - BENDRContextualizer: output shape
  - NEWBendingCollegeWav2Vec:
      - cluster contrastive loss (valid labels, all -1, mixed)
      - total loss includes InfoNCE + cluster terms
      - description string
      - memory bank update
      - reconstruction loss raises for GNN encoder (safety check)
"""

import pytest
import torch
import torch.nn as nn
from math import ceil
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dn3_ext import (
    ConvEncoderBENDR,
    BENDRContextualizer,
    NEWBendingCollegeWav2Vec,
)


# ── helpers ───────────────────────────────────────────────────────────────────

N_CH       = 21    # To1020 fixed channel count
ENCODER_H  = 64    # small for fast tests
SAMPLES    = 2560
BATCH      = 4


def make_encoder():
    return ConvEncoderBENDR(N_CH, encoder_h=ENCODER_H)


def make_context(encoder_h=ENCODER_H):
    return BENDRContextualizer(encoder_h, heads=8, layers=2, layer_drop=0.0)


def make_process(use_cluster=True, use_recon=False, num_clusters=3):
    enc  = make_encoder()
    ctx  = make_context()
    return NEWBendingCollegeWav2Vec(
        enc, ctx,
        use_cluster_loss=use_cluster,
        use_reconstruction_loss=use_recon,
        num_clusters=num_clusters,
        cluster_memory_size=50,
        num_negatives=5,
        mask_span=4,
    )


def encoded_len(samples=SAMPLES):
    # ask the actual encoder so the test always matches reality
    enc = make_encoder()
    return enc.downsampling_factor(samples)


def _device():
    """Use GPU if available — must match where DN3 puts the model weights."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── ConvEncoderBENDR ──────────────────────────────────────────────────────────

def test_conv_encoder_output_shape():
    enc = make_encoder()
    x   = torch.randn(BATCH, N_CH, SAMPLES)
    out = enc(x)
    expected_t = encoded_len()
    assert out.shape == (BATCH, ENCODER_H, expected_t), f"got {out.shape}"


def test_conv_encoder_output_dtype():
    enc = make_encoder()
    x   = torch.randn(2, N_CH, SAMPLES)
    assert enc(x).dtype == torch.float32


def test_conv_encoder_downsampling_factor():
    enc = make_encoder()
    # default strides (3,2,2,2,2,2) → product 48, ceil each step
    manual = SAMPLES
    for s in (3, 2, 2, 2, 2, 2):
        manual = ceil(manual / s)
    assert enc.downsampling_factor(SAMPLES) == manual


def test_conv_encoder_gradient_flows():
    enc = make_encoder()
    x   = torch.randn(2, N_CH, SAMPLES)
    enc(x).sum().backward()
    for name, p in enc.named_parameters():
        assert p.grad is not None, f"{name} no grad"


def test_conv_encoder_description_contains_hz():
    enc = make_encoder()
    desc = enc.description(sfreq=256, sequence_len=SAMPLES)
    assert "sfreq" in desc.lower() or "Hz" in desc


# ── BENDRContextualizer ───────────────────────────────────────────────────────

def test_contextualizer_output_shape():
    ctx  = make_context()
    T    = encoded_len()
    dev  = _device()
    ctx  = ctx.to(dev)
    z    = torch.randn(BATCH, ENCODER_H, T, device=dev)
    mask = torch.zeros(BATCH, T, dtype=torch.bool, device=dev)
    out  = ctx(z, mask)
    # contextualizer may adjust T slightly due to positional conv padding — just check B and H
    assert out.shape[0] == BATCH
    assert out.shape[1] == ENCODER_H


def test_contextualizer_with_partial_mask():
    ctx  = make_context()
    T    = encoded_len()
    dev  = _device()
    ctx  = ctx.to(dev)
    z    = torch.randn(BATCH, ENCODER_H, T, device=dev)
    mask = torch.zeros(BATCH, T, dtype=torch.bool, device=dev)
    mask[:, :T // 2] = True
    out  = ctx(z, mask)
    assert out.shape[0] == BATCH
    assert out.shape[1] == ENCODER_H


# ── NEWBendingCollegeWav2Vec: cluster loss ────────────────────────────────────

def test_cluster_loss_scalar():
    proc = make_process(use_cluster=True, num_clusters=3)
    T    = encoded_len()
    dev  = _device()
    emb  = torch.randn(BATCH, ENCODER_H, T, device=dev)
    labs = torch.zeros(BATCH, dtype=torch.long, device=dev)
    loss = proc._compute_cluster_contrastive_loss(emb, labs)
    assert loss.ndim == 0  # scalar tensor after squeeze
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_cluster_loss_all_transition_windows():
    proc = make_process(use_cluster=True)
    T    = encoded_len()
    dev  = _device()
    emb  = torch.randn(BATCH, ENCODER_H, T, device=dev)
    labs = torch.full((BATCH,), -1, dtype=torch.long, device=dev)
    loss = proc._compute_cluster_contrastive_loss(emb, labs)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_cluster_loss_mixed_labels():
    proc = make_process(use_cluster=True, num_clusters=3)
    T    = encoded_len()
    dev  = _device()
    emb  = torch.randn(BATCH, ENCODER_H, T, device=dev)
    labs = torch.tensor([0, 1, -1, 2], dtype=torch.long, device=dev)
    loss = proc._compute_cluster_contrastive_loss(emb, labs)
    assert not torch.isnan(loss)
    assert loss.item() >= 0.0


def test_cluster_loss_wraps_large_label():
    proc = make_process(use_cluster=True, num_clusters=3)
    T    = encoded_len()
    dev  = _device()
    emb  = torch.randn(BATCH, ENCODER_H, T, device=dev)
    labs = torch.tensor([5, 7, 10, 1], dtype=torch.long, device=dev)
    loss = proc._compute_cluster_contrastive_loss(emb, labs)
    assert not torch.isnan(loss)


def test_cluster_memory_updates():
    proc = make_process(use_cluster=True, num_clusters=2)
    T    = encoded_len()
    dev  = _device()
    emb  = torch.randn(BATCH, ENCODER_H, T, device=dev)
    labs = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=dev)
    # run once to let the memory bank migrate to the right device, then snapshot
    proc._compute_cluster_contrastive_loss(emb, labs)
    before = proc.cluster_ptr.clone()
    proc._compute_cluster_contrastive_loss(emb, labs)
    # both pointers must have advanced again after the second call
    assert (proc.cluster_ptr != before).any()


# ── NEWBendingCollegeWav2Vec: full forward + calculate_loss ──────────────────

def test_process_forward_shapes():
    proc = make_process(use_cluster=False)
    dev  = _device()
    # BaseProcess is not nn.Module — put encoder/context on device directly
    proc.encoder.to(dev)
    proc.context_fn.to(dev)
    x = torch.randn(BATCH, N_CH, SAMPLES, device=dev)
    logits, z, mask = proc.forward(x)
    assert z.shape[0]    == BATCH
    assert z.shape[1]    == ENCODER_H
    assert mask.shape[0] == BATCH


def test_process_calculate_loss_no_cluster():
    proc = make_process(use_cluster=False)
    dev  = _device()
    proc.encoder.to(dev)
    proc.context_fn.to(dev)
    x   = torch.randn(BATCH, N_CH, SAMPLES, device=dev)
    out = proc.forward(x)
    loss = proc.calculate_loss((x,), out)
    assert not torch.isnan(loss)
    assert loss.item() > 0


def test_process_calculate_loss_with_cluster():
    proc = make_process(use_cluster=True)
    dev  = _device()
    proc.encoder.to(dev)
    proc.context_fn.to(dev)
    x    = torch.randn(BATCH, N_CH, SAMPLES, device=dev)
    labs = torch.tensor([0, 1, 2, 0], dtype=torch.long, device=dev)
    out  = proc.forward(x, labs)
    loss = proc.calculate_loss((x, labs), out)
    assert not torch.isnan(loss)
    assert 'Cluster' in proc.loss_components


def test_process_loss_components_logged():
    proc = make_process(use_cluster=True)
    dev  = _device()
    proc.encoder.to(dev)
    proc.context_fn.to(dev)
    x    = torch.randn(BATCH, N_CH, SAMPLES, device=dev)
    labs = torch.zeros(BATCH, dtype=torch.long, device=dev)
    out  = proc.forward(x, labs)
    proc.calculate_loss((x, labs), out)
    assert 'InfoNCE' in proc.loss_components
    assert 'Cluster' in proc.loss_components


def test_process_description_string():
    proc = make_process(use_cluster=False)
    desc = proc.description(SAMPLES)
    assert isinstance(desc, str)
    assert len(desc) > 0


def test_process_cluster_loss_off_behaves_like_base():
    proc = make_process(use_cluster=False)
    dev  = _device()
    proc.encoder.to(dev)
    proc.context_fn.to(dev)
    x   = torch.randn(BATCH, N_CH, SAMPLES, device=dev)
    out = proc.forward(x)
    proc.calculate_loss((x,), out)
    assert 'Cluster' not in proc.loss_components


# ── reconstruction loss safety check ─────────────────────────────────────────

def test_reconstruction_loss_raises_for_non_conv_encoder():
    class FakeEncoder(nn.Module):
        encoder_h = ENCODER_H
        def forward(self, x): return x
        def downsampling_factor(self, s): return s // 48

    with pytest.raises(ValueError, match="ConvEncoderBENDR"):
        NEWBendingCollegeWav2Vec(
            FakeEncoder(), make_context(),
            use_cluster_loss=False,
            use_reconstruction_loss=True,
        )
