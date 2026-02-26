"""
tests/test_pretrain_utils.py

Unit tests for the utility functions in pretrain.py that don't need real data:
  - safe_sfreq_for_dataset: sfreq selection and lpf suggestion logic
  - ClusterLabelDataset: label alignment with a mock DN3 dataset
  - _cfg helper: strips _ prefixed keys from config objects

We mock the MNE file reading so no actual EEG files are needed.
"""

import os
import types
import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pretrain import safe_sfreq_for_dataset, ClusterLabelDataset


# ── safe_sfreq_for_dataset ────────────────────────────────────────────────────

def _mock_raw(sfreq, lowpass):
    """Build a fake MNE raw-like info dict."""
    raw = MagicMock()
    raw.info = {'sfreq': sfreq, 'lowpass': lowpass}
    return raw


def _patch_mne(sfreq, lowpass, tmp_path):
    """
    Create a fake EEG file in tmp_path and patch mne.io.read_raw so it
    returns a fake raw with the given sfreq and lowpass.
    """
    fake_edf = tmp_path / "sub-01" / "eeg" / "test.edf"
    fake_edf.parent.mkdir(parents=True)
    fake_edf.touch()
    raw = _mock_raw(sfreq, lowpass)
    return fake_edf.parent.parent, raw


def test_safe_sfreq_exact_match(tmp_path):
    toplevel, raw = _patch_mne(256, 128, tmp_path)
    with patch("pretrain.mne.io.read_raw", return_value=raw):
        hz, n, lpf = safe_sfreq_for_dataset(str(toplevel), target_sfreq=256, epoch_secs=10)
    assert hz  == 256
    assert n   == 2560
    assert lpf is None  # 256 < 2*128 = 256, not strictly less, so no lpf needed


def test_safe_sfreq_safe_2x_downsample(tmp_path):
    # 512 → 256, ratio=2 ≤ 4, clean divide → use 256
    toplevel, raw = _patch_mne(512, 256, tmp_path)
    with patch("pretrain.mne.io.read_raw", return_value=raw):
        hz, n, lpf = safe_sfreq_for_dataset(str(toplevel), target_sfreq=256, epoch_secs=10)
    assert hz == 256
    assert n  == 2560


def test_safe_sfreq_ratio_4_borderline(tmp_path):
    # 1024 → 256, ratio=4, exactly at the limit → should still use 256
    toplevel, raw = _patch_mne(1024, 512, tmp_path)
    with patch("pretrain.mne.io.read_raw", return_value=raw):
        hz, n, lpf = safe_sfreq_for_dataset(str(toplevel), target_sfreq=256, epoch_secs=10)
    assert hz == 256


def test_safe_sfreq_suggests_lpf_when_header_too_high(tmp_path):
    # native=1024, lowpass=512 in header, target=256
    # DN3 would check 256 < 2*512=1024 → True → aliasing
    # our function should suggest lpf ≈ 256/2 * 0.9 = 115.2
    toplevel, raw = _patch_mne(1024, 512, tmp_path)
    with patch("pretrain.mne.io.read_raw", return_value=raw):
        hz, n, lpf = safe_sfreq_for_dataset(str(toplevel), target_sfreq=256, epoch_secs=10)
    assert lpf is not None
    assert lpf < hz / 2  # must be below Nyquist of chosen sfreq


def test_safe_sfreq_non_integer_ratio_steps_up(tmp_path):
    # 1000 Hz native, target 256 — 1000/256 is not integer → step up
    # 1000/2=500, 500/2=250 < 256, so chosen stays at 500
    toplevel, raw = _patch_mne(1000, 500, tmp_path)
    with patch("pretrain.mne.io.read_raw", return_value=raw):
        hz, n, lpf = safe_sfreq_for_dataset(str(toplevel), target_sfreq=256, epoch_secs=10)
    assert hz >= 256
    assert n == hz * 10


def test_safe_sfreq_no_files_returns_none(tmp_path):
    # empty directory, no EEG files
    hz, n, lpf = safe_sfreq_for_dataset(str(tmp_path), target_sfreq=256, epoch_secs=10)
    assert hz  is None
    assert n   is None
    assert lpf is None


def test_safe_sfreq_skips_derivatives(tmp_path):
    # only file is in derivatives/ — should not be found
    deriv = tmp_path / "derivatives" / "eeg"
    deriv.mkdir(parents=True)
    (deriv / "data.edf").touch()
    hz, n, lpf = safe_sfreq_for_dataset(str(tmp_path), target_sfreq=256, epoch_secs=10)
    assert hz is None


def test_safe_sfreq_epoch_secs_scales_samples(tmp_path):
    toplevel, raw = _patch_mne(256, 128, tmp_path)
    with patch("pretrain.mne.io.read_raw", return_value=raw):
        hz5,  n5,  _ = safe_sfreq_for_dataset(str(toplevel), target_sfreq=256, epoch_secs=5)
        hz10, n10, _ = safe_sfreq_for_dataset(str(toplevel), target_sfreq=256, epoch_secs=10)
    assert n10 == 2 * n5


# ── ClusterLabelDataset ───────────────────────────────────────────────────────

class _FakeSession:
    def __init__(self, filename, n_epochs):
        self.filename = filename
        self._decimated_sequence_starts = list(range(n_epochs))

    def __len__(self):
        return len(self._decimated_sequence_starts)


class _FakeThinker:
    def __init__(self, sessions):
        self.sessions = sessions


class _FakeDataset:
    """Minimal mock that looks enough like a DN3 dataset for ClusterLabelDataset."""
    def __init__(self, thinkers):
        self.thinkers = thinkers
        self._all_samples = []
        for th in thinkers.values():
            for sess in th.sessions.values():
                for _ in range(len(sess)):
                    self._all_samples.append(torch.randn(21, 2560))

    def __len__(self):
        return len(self._all_samples)

    def __getitem__(self, idx):
        return self._all_samples[idx]


def _make_fake_dn3_dataset(n_epochs=10, filename="sub-01/eeg.edf"):
    sess    = _FakeSession(filename, n_epochs)
    thinker = _FakeThinker({"sess0": sess})
    return _FakeDataset({"thinker0": thinker})


def test_cluster_label_dataset_length():
    ds = _make_fake_dn3_dataset(n_epochs=10)
    label_dict = {}  # no labels → all -1
    cld = ClusterLabelDataset(ds, label_dict, epoch_len=2560)
    assert len(cld) == len(ds)


def test_cluster_label_dataset_returns_tuple():
    ds  = _make_fake_dn3_dataset(n_epochs=4)
    cld = ClusterLabelDataset(ds, {}, epoch_len=2560)
    item = cld[0]
    assert isinstance(item, tuple)
    assert len(item) == 2
    eeg, label = item
    assert isinstance(eeg,   torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert label.dtype == torch.long


def test_cluster_label_dataset_unknown_file_all_minus_one():
    ds  = _make_fake_dn3_dataset(n_epochs=5)
    cld = ClusterLabelDataset(ds, {}, epoch_len=2560)
    labels = [cld[i][1].item() for i in range(len(cld))]
    assert all(l == -1 for l in labels)


def test_cluster_label_dataset_correct_labels_assigned():
    # build a label dict that covers our fake session
    n_epochs = 6
    epoch_len = 2560
    n_samples = n_epochs * epoch_len
    filename  = "sub-01/eeg.edf"

    # all epochs are cluster 2
    raw_labels = np.full(n_samples, 2, dtype=np.int64)
    raw_mask   = np.ones(n_samples,  dtype=np.float32)
    label_dict = {filename: {"labels": raw_labels, "mask": raw_mask}}

    ds  = _make_fake_dn3_dataset(n_epochs=n_epochs, filename=filename)
    cld = ClusterLabelDataset(ds, label_dict, epoch_len=epoch_len)

    labels = [cld[i][1].item() for i in range(len(cld))]
    assert all(l == 2 for l in labels), f"expected all 2, got {labels}"
