"""
tests/test_gen_labels.py

Unit tests for gen_labels.py — label creation, epoch collapsing, JSON parsing,
and the full save/load round-trip.
"""

import json
import tempfile
import numpy as np
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from gen_labels import (
    create_labels,
    epoch_label,
    extract_switches,
    build_epoch_label_array,
    gen_label_vec,
    load_label_file,
)

SFREQ    = 256
EPOCH    = SFREQ * 10  # 2560 samples


# ── create_labels ────────────────────────────────────────────────────────────

def test_create_labels_no_switches():
    # one continuous stable segment — whole recording is one task (id 0)
    label, mask = create_labels([], total_samples=EPOCH)
    assert label.shape == (EPOCH,)
    assert mask.shape  == (EPOCH,)
    assert (label == 0).all()
    assert (mask  == 1.0).all()


def test_create_labels_single_switch():
    # one transition at [500, 700]: stable 0→500, transition 500→700, stable 700→end
    label, mask = create_labels([(500, 700)], total_samples=1000)
    assert (label[:500] == 0).all(),   "before transition should be cluster 0"
    assert (label[500:700] == -1).all(), "transition window should be -1"
    assert (label[700:] == 1).all(),   "after transition should be cluster 1"
    assert (mask[:500]  == 1.0).all()
    assert (mask[500:700] == 0.0).all()
    assert (mask[700:]  == 1.0).all()


def test_create_labels_two_switches():
    # two transitions: [100,150] and [400,450]
    label, mask = create_labels([(100, 150), (400, 450)], total_samples=600)
    assert (label[:100]   == 0).all()
    assert (label[100:150]== -1).all()
    assert (label[150:400]== 1).all()
    assert (label[400:450]== -1).all()
    assert (label[450:]   == 2).all()


def test_create_labels_output_dtypes():
    label, mask = create_labels([(50, 100)], total_samples=200)
    assert label.dtype == np.int64
    assert mask.dtype  == np.float32


def test_create_labels_switch_at_very_start():
    # transition starts at 0 — no stable region before it
    label, mask = create_labels([(0, 50)], total_samples=200)
    assert (label[0:50] == -1).all()
    assert (label[50:]  == 1).all()


# ── epoch_label ──────────────────────────────────────────────────────────────

def test_epoch_label_clean_epoch():
    # all samples in this epoch are cluster 2
    label = np.full(EPOCH, 2, dtype=np.int64)
    mask  = np.ones(EPOCH,  dtype=np.float32)
    assert epoch_label(label, mask, epoch_start=0) == 2


def test_epoch_label_majority_vote():
    # first 60% cluster 0, last 40% cluster 1 → 0 wins
    label = np.zeros(EPOCH, dtype=np.int64)
    label[int(0.6 * EPOCH):] = 1
    mask  = np.ones(EPOCH, dtype=np.float32)
    assert epoch_label(label, mask, epoch_start=0) == 0


def test_epoch_label_transition_majority_returns_minus_one():
    # more than half of the epoch is a transition window
    label = np.full(EPOCH, -1, dtype=np.int64)
    mask  = np.zeros(EPOCH, dtype=np.float32)
    assert epoch_label(label, mask, epoch_start=0) == -1


def test_epoch_label_partial_transition_threshold():
    # less than 50% usable → should return -1
    label = np.zeros(EPOCH, dtype=np.int64)
    mask  = np.zeros(EPOCH, dtype=np.float32)
    mask[:EPOCH // 2 - 1] = 1.0  # just under 50%
    assert epoch_label(label, mask, epoch_start=0) == -1


def test_epoch_label_offset_start():
    # make sure epoch_start slice works correctly
    label = np.full(2 * EPOCH, -1, dtype=np.int64)
    mask  = np.zeros(2 * EPOCH, dtype=np.float32)
    # only the second epoch is clean cluster 3
    label[EPOCH:] = 3
    mask[EPOCH:]  = 1.0
    assert epoch_label(label, mask, epoch_start=EPOCH) == 3


# ── extract_switches ──────────────────────────────────────────────────────────

def test_extract_switches_parses_json(tmp_path):
    data = [
        {"name": "sub-01/eeg.edf", "task_switch": [[100, 200], [500, 600]]},
        {"name": "sub-02/eeg.edf", "task_switch": []},
    ]
    p = tmp_path / "switches.json"
    p.write_text(json.dumps(data))

    result = extract_switches(str(p))
    assert set(result.keys()) == {"sub-01/eeg.edf", "sub-02/eeg.edf"}
    assert result["sub-01/eeg.edf"] == [(100, 200), (500, 600)]
    assert result["sub-02/eeg.edf"] == []


def test_extract_switches_tuples_not_lists(tmp_path):
    data = [{"name": "f.edf", "task_switch": [[10, 20]]}]
    p = tmp_path / "sw.json"
    p.write_text(json.dumps(data))
    result = extract_switches(str(p))
    assert isinstance(result["f.edf"][0], tuple)


# ── build_epoch_label_array ───────────────────────────────────────────────────

def test_build_epoch_label_array_length():
    # with 3 * EPOCH samples and stride = EPOCH we expect 3 epochs
    n = 3 * EPOCH
    label = np.zeros(n, dtype=np.int64)
    mask  = np.ones(n,  dtype=np.float32)
    ld = {"file.edf": {"labels": label, "mask": mask}}
    arr = build_epoch_label_array(ld, "file.edf", n_samples=n, epoch_len=EPOCH)
    assert len(arr) == 3


def test_build_epoch_label_array_missing_file():
    # file not in label dict → all -1, length still correct
    arr = build_epoch_label_array({}, "missing.edf", n_samples=EPOCH * 5, epoch_len=EPOCH)
    assert (arr == -1).all()
    assert len(arr) == 5


# ── gen_label_vec + load_label_file (round-trip) ─────────────────────────────

def test_gen_and_load_round_trip(tmp_path):
    data = [{"name": "sub/rec.edf", "task_switch": [[500, 700]]}]
    sw_file  = tmp_path / "sw.json"
    out_file = tmp_path / "labels.npy"
    sw_file.write_text(json.dumps(data))

    results = gen_label_vec(str(sw_file), str(out_file),
                             total_samples_map={"sub/rec.edf": 2560})

    loaded = load_label_file(str(out_file))

    assert "sub/rec.edf" in loaded
    assert np.array_equal(loaded["sub/rec.edf"]["labels"],
                           results["sub/rec.edf"]["labels"])
    assert np.array_equal(loaded["sub/rec.edf"]["mask"],
                           results["sub/rec.edf"]["mask"])
