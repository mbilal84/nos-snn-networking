# src/metrics.py
"""
Metrics and small analysis utilities used across NOS experiments.

Includes:
- Empirical CCDF for nonnegative samples.
- Spike train rate and ISI CV.
- Simple event detection from residual z-scores and matching utilities
  (kept minimal, so scripts can choose stricter definitions if desired).

Corner-case conventions:
- Some metrics (e.g., match_event_starts) define explicit return values when one or both
  event lists are empty. These are documented in the respective docstrings so evaluation
  scripts remain deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def empirical_ccdf(x: np.ndarray, kind: str = "geq") -> Tuple[np.ndarray, np.ndarray]:
    """
    Empirical CCDF for a 1D sample array x.

    Returns (xs, p) where xs is sorted unique values (or sorted samples),
    and p approximates P[X >= xs] or P[X > xs].

    kind:
      - "geq": P[X >= x]
      - "gt":  P[X > x]

    Note: For plots, using sorted samples (not unique bins) is fine.
    The returned xs are the sorted finite samples (same length as the filtered input),
    so duplicate x-values are expected. To get unique x-values, post-process with
    np.unique(xs, return_index=True) and take p at the corresponding indices.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([]), np.array([])

    xs = np.sort(x)
    n = xs.size

    if kind == "geq":
        # At sorted sample xs[i], probability mass at and above that index
        # P[X >= xs[i]] ≈ (n - i) / n
        p = (n - np.arange(n)) / n
    elif kind == "gt":
        # P[X > xs[i]] ≈ (n - i - 1) / n, clipped at 0
        p = (n - np.arange(n) - 1) / n
        p = np.clip(p, 0.0, 1.0)
    else:
        raise ValueError("kind must be 'geq' or 'gt'")

    return xs, p


def firing_rate(spikes: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute per-neuron firing rate from a (T,N) spike array (bool or 0/1).

    Returns rate (N,) in Hz if dt is seconds.
    """
    S = np.asarray(spikes)
    if S.ndim != 2:
        raise ValueError("spikes must have shape (T,N)")
    if dt <= 0:
        raise ValueError("dt must be positive")
    return S.sum(axis=0) / (S.shape[0] * dt)


def isi_cv(spikes: np.ndarray, dt: float, min_spikes: int = 5) -> np.ndarray:
    """
    Compute ISI coefficient of variation per neuron from (T,N) spikes.

    Returns CV (N,), with NaN for neurons with < min_spikes spikes.
    """
    S = np.asarray(spikes).astype(bool)
    if S.ndim != 2:
        raise ValueError("spikes must have shape (T,N)")
    if dt <= 0:
        raise ValueError("dt must be positive")

    T, N = S.shape
    out = np.full(N, np.nan, dtype=float)
    for i in range(N):
        idx = np.where(S[:, i])[0]
        if idx.size < min_spikes:
            continue
        isi = np.diff(idx) * dt
        m = float(np.mean(isi))
        sd = float(np.std(isi, ddof=1)) if isi.size > 1 else 0.0
        if m > 0:
            out[i] = sd / m
    return out


# ----------------------------
# Residual-event utilities
# --------------------------

@dataclass(frozen=True)
class RobustStats:
    median: float
    mad: float  # median absolute deviation (scaled or unscaled, caller decides)

    def z(self, x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        return (x - self.median) / (self.mad + eps)


def robust_median_mad(x: np.ndarray, scale_mad_to_sigma: bool = True) -> RobustStats:
    """
    Robust centre/scale using median and MAD.

    If scale_mad_to_sigma=True, uses 1.4826*MAD to approximate sigma for Gaussian.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return RobustStats(median=0.0, mad=1.0)

    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if scale_mad_to_sigma:
        mad = 1.4826 * mad
    if mad <= 1e-15:
        mad = 1.0
    return RobustStats(median=med, mad=mad)


def detect_events_from_residuals(
    residual: np.ndarray,
    stats: RobustStats,
    z_thresh: float = 3.0,
    min_len: int = 1,
    polarity: str = "positive",
) -> List[Tuple[int, int]]:
    """
    Convert residual time series into event intervals based on robust z-scores.

    residual: shape (T,)
    polarity:
      - "positive": events where z >= z_thresh
      - "negative": events where z <= -z_thresh
      - "both": events where |z| >= z_thresh

    Returns list of (start_idx, end_idx) inclusive.
    Indices are in sample steps (0..T-1). 
    """
    r = np.asarray(residual, dtype=float).ravel()
    z = stats.z(r)

    if polarity == "positive":
        mask = z >= z_thresh
    elif polarity == "negative":
        mask = z <= -z_thresh
    elif polarity == "both":
        mask = np.abs(z) >= z_thresh
    else:
        raise ValueError("polarity must be 'positive', 'negative', or 'both'")

    events: List[Tuple[int, int]] = []
    if mask.size == 0:
        return events

    in_evt = False
    s = 0
    for i, m in enumerate(mask):
        if m and not in_evt:
            in_evt = True
            s = i
        elif (not m) and in_evt:
            e = i - 1
            if (e - s + 1) >= min_len:
                events.append((s, e))
            in_evt = False
    if in_evt:
        e = mask.size - 1
        if (e - s + 1) >= min_len:
            events.append((s, e))
    return events


def match_event_starts(
    truth_events: List[Tuple[int, int]],
    pred_events: List[Tuple[int, int]],
    window: int = 10,
    allow_early: bool = False,
) -> Dict[str, float]:
    """
    Match predicted event starts to truth event starts within a window.

    If allow_early=False:
        pred_start is a hit if truth_start <= pred_start <= truth_start + window
        (early warnings count as false positives)
    If allow_early=True:
        pred_start is a hit if truth_start - window <= pred_start <= truth_start + window

    Returns dict with precision, recall, f1, mean_latency (signed).
Empty-list conventions:
    - If truth_events and pred_events are both empty: precision=recall=f1=1.0 and mean_latency=0.0.
    - If truth_events is empty but pred_events is not: precision=0.0, recall=1.0, f1=0.0.
    - If pred_events is empty but truth_events is not: precision=1.0, recall=0.0, f1=0.0.
These conventions avoid divide-by-zero and keep evaluation deterministic; if we prefer
undefined metrics in these cases, wrap this function and return NaN instead.
    Latency = pred_start - truth_start, averaged over matched events.
    """
    t_starts = [s for (s, _) in truth_events]
    p_starts = [s for (s, _) in pred_events]

    if len(t_starts) == 0 and len(p_starts) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "mean_latency": 0.0}
    if len(t_starts) == 0:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "mean_latency": float("nan")}
    if len(p_starts) == 0:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0, "mean_latency": float("nan")}

    t_used = np.zeros(len(t_starts), dtype=bool)
    hits = 0
    latencies: List[int] = []

    for ps in p_starts:
        # Find the first unused truth start that matches
        for ti, ts in enumerate(t_starts):
            if t_used[ti]:
                continue
            if allow_early:
                ok = (ts - window) <= ps <= (ts + window)
            else:
                ok = ts <= ps <= (ts + window)
            if ok:
                t_used[ti] = True
                hits += 1
                latencies.append(ps - ts)
                break

    precision = hits / len(p_starts)
    recall = hits / len(t_starts)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_latency = float(np.mean(latencies)) if latencies else float("nan")

    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "mean_latency": mean_latency}
