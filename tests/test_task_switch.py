"""
tests/test_task_switch.py

Unit tests for the signal-processing utilities in task_switch_identifi.py.

We don't run the full pipeline (that needs real EEG files and hours of compute),
but we test every math function in isolation with synthetic data so we know the
building blocks are correct before spending time on real data.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from task_switch_identifi import (
    bhattacharyya_distance,
    butter_bandpass_filter,
    get_combined_envelope_sq,
    calculate_GFP,
    get_multi_channel_band_power,
    verify_alpha_theta_2,
    gfp_check,
)

SFREQ = 256
N_CH  = 8
T     = SFREQ * 20  # 20 seconds of synthetic data


# ── helpers ───────────────────────────────────────────────────────────────────

def make_eeg_df(n_ch=N_CH, t=T, sfreq=SFREQ, seed=42):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((t, n_ch)).astype(np.float32)
    return pd.DataFrame(data, columns=[f"ch{i}" for i in range(n_ch)])


def make_sine(freq_hz, t=T, sfreq=SFREQ, n_ch=N_CH, amplitude=1.0):
    """Pure sine wave for testing band-power functions."""
    ts = np.arange(t) / sfreq
    wave = amplitude * np.sin(2 * np.pi * freq_hz * ts)
    data = np.tile(wave[:, None], (1, n_ch)).astype(np.float32)
    return pd.DataFrame(data, columns=[f"ch{i}" for i in range(n_ch)])


# ── bhattacharyya_distance ────────────────────────────────────────────────────

def test_bhattacharyya_identical_distributions():
    psd = np.array([1.0, 2.0, 3.0, 4.0])
    d = bhattacharyya_distance(psd, psd)
    # BC of identical normalized distributions = 1 → -log(1+1e-10) ≈ 0
    assert d < 0.01, f"expected ~0, got {d}"


def test_bhattacharyya_orthogonal_distributions():
    # disjoint support → BC = 0 → distance = -log(0 + 1e-10) ≈ large
    psd1 = np.array([1.0, 0.0, 0.0])
    psd2 = np.array([0.0, 1.0, 0.0])
    d = bhattacharyya_distance(psd1, psd2)
    assert d > 5, f"expected large distance for orthogonal PSDs, got {d}"


def test_bhattacharyya_nonnegative():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p1 = np.abs(rng.standard_normal(32)) + 1e-6
        p2 = np.abs(rng.standard_normal(32)) + 1e-6
        assert bhattacharyya_distance(p1, p2) >= 0


def test_bhattacharyya_symmetric():
    p1 = np.array([1.0, 2.0, 3.0])
    p2 = np.array([3.0, 2.0, 1.0])
    d12 = bhattacharyya_distance(p1, p2)
    d21 = bhattacharyya_distance(p2, p1)
    assert abs(d12 - d21) < 1e-6, "bhattacharyya distance should be symmetric"


# ── butter_bandpass_filter ────────────────────────────────────────────────────

def test_bandpass_output_shape():
    data = np.random.randn(T, N_CH).astype(np.float32)
    out  = butter_bandpass_filter(data, low=8, high=13, sfreq=SFREQ)
    assert out.shape == data.shape


def test_bandpass_attenuates_out_of_band():
    # 10 Hz alpha tone should survive 8-13 Hz filter, 40 Hz should be gone
    alpha_df = make_sine(freq_hz=10.0)
    gamma_df = make_sine(freq_hz=40.0)

    alpha_data = alpha_df.values.astype(np.float32)
    gamma_data = gamma_df.values.astype(np.float32)

    alpha_filt = butter_bandpass_filter(alpha_data, 8, 13, sfreq=SFREQ)
    gamma_filt = butter_bandpass_filter(gamma_data, 8, 13, sfreq=SFREQ)

    alpha_power = np.var(alpha_filt)
    gamma_power = np.var(gamma_filt)
    assert alpha_power > 100 * gamma_power, (
        f"alpha power {alpha_power:.3f} should >> attenuated gamma {gamma_power:.3f}")


# ── get_multi_channel_band_power ──────────────────────────────────────────────

def test_band_power_positive():
    df  = make_eeg_df()
    val = get_multi_channel_band_power(df.values.T, sfreq=SFREQ, band=(8, 13))
    assert val > 0


def test_band_power_alpha_sine_higher_than_noise_at_other_band():
    # a pure alpha tone should have higher alpha power than gamma power
    df = make_sine(freq_hz=10.0)
    alpha_p = get_multi_channel_band_power(df.values.T, sfreq=SFREQ, band=(8, 13))
    gamma_p = get_multi_channel_band_power(df.values.T, sfreq=SFREQ, band=(30, 45))
    assert alpha_p > gamma_p


# ── get_combined_envelope_sq ──────────────────────────────────────────────────

def test_envelope_sq_shape():
    df  = make_eeg_df()
    env = get_combined_envelope_sq(df, low=8, high=13, fs=SFREQ)
    assert env.shape == (T,)


def test_envelope_sq_nonnegative():
    df  = make_eeg_df()
    env = get_combined_envelope_sq(df, low=8, high=13, fs=SFREQ)
    assert (env >= 0).all()


def test_envelope_sq_higher_for_strong_alpha():
    # pure alpha sine → envelope should be much larger than noise envelope
    alpha_df = make_sine(freq_hz=10.0, amplitude=5.0)
    noise_df = make_eeg_df()

    env_alpha = get_combined_envelope_sq(alpha_df, low=8, high=13, fs=SFREQ)
    env_noise = get_combined_envelope_sq(noise_df, low=8, high=13, fs=SFREQ)
    assert env_alpha.mean() > env_noise.mean()


# ── calculate_GFP ─────────────────────────────────────────────────────────────

def test_gfp_shape():
    df  = make_eeg_df()
    gfp = calculate_GFP(df, sfreq=SFREQ)
    assert gfp.shape == (T,)


def test_gfp_nonnegative():
    df  = make_eeg_df()
    gfp = calculate_GFP(df, sfreq=SFREQ)
    assert (gfp >= 0).all()


def test_gfp_zero_for_constant_signal():
    # all channels identical → std across channels = 0 → GFP = 0
    const = np.ones((T, N_CH), dtype=np.float32)
    df    = pd.DataFrame(const, columns=[f"ch{i}" for i in range(N_CH)])
    gfp   = calculate_GFP(df, sfreq=SFREQ)
    assert np.allclose(gfp, 0, atol=1e-5)


def test_gfp_increases_with_variance():
    low_var  = make_eeg_df(seed=0)  * 0.01
    high_var = make_eeg_df(seed=1)  * 10.0
    assert calculate_GFP(high_var, sfreq=SFREQ).mean() > \
           calculate_GFP(low_var,  sfreq=SFREQ).mean()


# ── verify_alpha_theta_2 ──────────────────────────────────────────────────────

def test_verify_alpha_theta_no_candidates_returns_empty():
    df = make_eeg_df()
    result = verify_alpha_theta_2(df, candidate_time_points=[], sfreq=SFREQ)
    assert result == []


def test_verify_alpha_theta_returns_list():
    df = make_eeg_df()
    # pass a few candidate points; may or may not validate — just check type
    candidates = [SFREQ * 2, SFREQ * 5, SFREQ * 10]
    result = verify_alpha_theta_2(df, candidate_time_points=candidates, sfreq=SFREQ)
    assert isinstance(result, list)
    for seg in result:
        assert isinstance(seg, tuple)
        assert len(seg) == 2


# ── gfp_check ─────────────────────────────────────────────────────────────────

def test_gfp_check_no_candidates():
    df  = make_eeg_df()
    gfp = calculate_GFP(df, sfreq=SFREQ)
    result = gfp_check([], gfp, sfreq=SFREQ)
    assert result == []


def test_gfp_check_returns_list():
    df  = make_eeg_df()
    gfp = calculate_GFP(df, sfreq=SFREQ)
    # a transition at the middle of the signal
    mid  = T // 2
    segs = [(mid, mid + int(0.2 * SFREQ))]
    result = gfp_check(segs, gfp, sfreq=SFREQ)
    assert isinstance(result, list)


def test_gfp_check_filters_early_segments():
    # segments that start before 1500 ms should always be skipped
    df  = make_eeg_df()
    gfp = calculate_GFP(df, sfreq=SFREQ)
    early_seg = [(10, 50)]  # way before 1500ms
    result = gfp_check(early_seg, gfp, sfreq=SFREQ)
    assert result == []
