# src/topology.py
"""
Topology utilities for NOS experiments.

What this module provides:
- Simple graph constructors: chain, star, scale-free (Barabasi-Albert).
- Lognormal edge weights (symmetric or directed).
- Optional spectral-radius normalisation.
- Optional integer delay matrix (steps) for event-delayed coupling.

Keep all W/delay generation here so every experiment is consistent.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple, Literal

import numpy as np


TopologyKind = Literal["chain", "star", "scale_free"]


@dataclass(frozen=True)
class TopologySpec:
    kind: TopologyKind
    N: int

    # For scale-free
    m: int = 3                # edges per new node

    # Symmetry / direction
    symmetrise: bool = True   # enforce W = W^T after weighting
    directed: bool = False    # if True, weights are per-direction (ignored if symmetrise=True)
                             # Note: built-in adjacencies are undirected; 'directed' affects weighting only.

    # Weight distribution (lognormal)
    w_lognorm_mean: float = 0.0
    w_lognorm_sigma: float = 0.5

    # Spectral radius scaling (None means no scaling)
    target_rho: Optional[float] = 1.0

    # Randomness
    seed: int = 1


def spectral_radius(W: np.ndarray, max_iter: int = 200, tol: float = 1e-10) -> float:
    """
    Estimate spectral radius rho(W) = max |eig(W)|.
    Uses exact eigvals for small matrices; power iteration otherwise.

    Note: for non-normal matrices power iteration targets the dominant eigenvalue
    in magnitude if it is well separated. This is usually good enough for scaling.
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be square (N,N).")

    N = W.shape[0]
    if N <= 256:
        vals = np.linalg.eigvals(W)
        return float(np.max(np.abs(vals)))

    rng = np.random.default_rng(0)
    x = rng.normal(size=N)
    x = x / (np.linalg.norm(x) + 1e-12)

    last = 0.0
    for _ in range(max_iter):
        y = W @ x
        ny = float(np.linalg.norm(y))
        if ny < 1e-15:
            return 0.0
        x = y / ny
        if abs(ny - last) <= tol * max(1.0, ny):
            last = ny
            break
        last = ny

    # Rayleigh quotient on the final iterate (still only an estimate)
    y = W @ x
    rq = float(np.dot(x, y))
    return float(max(abs(rq), last))


def _adj_chain(N: int) -> np.ndarray:
    A = np.zeros((N, N), dtype=np.uint8)
    for i in range(N - 1):
        A[i, i + 1] = 1
        A[i + 1, i] = 1
    return A


def _adj_star(N: int, center: int = 0) -> np.ndarray:
    A = np.zeros((N, N), dtype=np.uint8)
    for i in range(N):
        if i == center:
            continue
        A[center, i] = 1
        A[i, center] = 1
    return A


def _adj_scale_free_ba(N: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """
    Simple Barabasi-Albert undirected adjacency, without external dependencies.

    Starts from a (m+1)-clique, then adds nodes with preferential attachment.
    """
    if m < 1:
        raise ValueError("m must be >= 1")
    if N < m + 2:
        raise ValueError("For BA, require N >= m+2")

    A = np.zeros((N, N), dtype=np.uint8)

    m0 = m + 1
    # Initial clique
    for i in range(m0):
        for j in range(i + 1, m0):
            A[i, j] = 1
            A[j, i] = 1

    deg = A.sum(axis=1).astype(float)  # degree sequence

    # Preferential attachment
    for new in range(m0, N):
        probs = deg[:new].copy()
        s = probs.sum()
        if s <= 0:
            probs[:] = 1.0 / new
        else:
            probs /= s

        # Choose m distinct targets
        targets = rng.choice(new, size=m, replace=False, p=probs)
        for t in targets:
            A[new, t] = 1
            A[t, new] = 1

        deg[new] = m
        deg[targets] += 1.0

    return A


def apply_lognormal_weights(
    A: np.ndarray,
    rng: np.random.Generator,
    mean: float,
    sigma: float,
    symmetrise: bool = True,
    directed: bool = False,
) -> np.ndarray:
    """
    Turn a 0/1 adjacency matrix A into a weighted matrix W using lognormal weights.

    If symmetrise=True: sample one weight per undirected edge and mirror it.
    If symmetrise=False and directed=True: sample per directed edge (A[i,j]=1).
    If symmetrise=False and directed=False: still uses A as given; weights per edge.
    """
    A = np.asarray(A, dtype=np.uint8)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square (N,N).")

    N = A.shape[0]
    W = np.zeros((N, N), dtype=float)

    if symmetrise:
        # Use union of edges and mirror weights.
        for i in range(N):
            for j in range(i + 1, N):
                if A[i, j] or A[j, i]:
                    w = float(rng.lognormal(mean=mean, sigma=sigma))
                    W[i, j] = w
                    W[j, i] = w
    else:
        if directed:
            idx = np.argwhere(A != 0)
            for i, j in idx:
                if i == j:
                    continue
                W[i, j] = float(rng.lognormal(mean=mean, sigma=sigma))
        else:
            # Use A as provided; treat nonzeros as edges (could be asymmetric).
            idx = np.argwhere(A != 0)
            for i, j in idx:
                if i == j:
                    continue
                W[i, j] = float(rng.lognormal(mean=mean, sigma=sigma))

    np.fill_diagonal(W, 0.0)
    return W


def normalise_to_spectral_radius(W: np.ndarray, target_rho: float) -> Tuple[np.ndarray, float]:
    """
    Scale W so that spectral_radius(W) equals target_rho, if rho>0.
    Returns (W_scaled, rho_before).
    """
    W = np.asarray(W, dtype=float)
    rho = spectral_radius(W)
    if rho <= 0.0:
        return W.copy(), float(rho)
    return (target_rho / rho) * W, float(rho)


def build_topology(spec: TopologySpec) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build W according to spec, and return (W, meta).
    """
    if spec.N <= 1:
        raise ValueError("N must be >= 2")

    rng = np.random.default_rng(spec.seed)

    if spec.kind == "chain":
        A = _adj_chain(spec.N)
    elif spec.kind == "star":
        A = _adj_star(spec.N, center=0)
    elif spec.kind == "scale_free":
        A = _adj_scale_free_ba(spec.N, m=spec.m, rng=rng)
    else:
        raise ValueError(f"Unknown kind: {spec.kind}")

    W = apply_lognormal_weights(
        A=A,
        rng=rng,
        mean=spec.w_lognorm_mean,
        sigma=spec.w_lognorm_sigma,
        symmetrise=spec.symmetrise,
        directed=spec.directed,
    )

    rho_before_scale = spectral_radius(W)
    rho_target = spec.target_rho
    if rho_target is not None:
        W, rho_before = normalise_to_spectral_radius(W, float(rho_target))
        rho_after = spectral_radius(W)
    else:
        rho_before = rho_before_scale
        rho_after = rho_before_scale

    meta: Dict[str, Any] = spec_to_meta(spec)
    meta.update(
        {
            "rho_before": float(rho_before),
            "rho_after": float(rho_after),
        }
    )
    return W, meta


def build_delays_steps(
    W: np.ndarray,
    dt: float,
    delay_range_s: Tuple[float, float] = (0.0, 0.05),
    rng: Optional[np.random.Generator] = None,
    include_self: bool = False,
) -> np.ndarray:
    """
    Build integer delay matrix (steps) for each nonzero edge in W.

    delay_range_s: (min_delay_seconds, max_delay_seconds)
Notes:
    - Delays are quantised to integer steps using round(delay_s / dt). This means very small
      delays may become 0 steps (same-call delivery in EventDelayCoupling.step()).
    - If rng is None, a fixed seed is used so delay matrices are reproducible by default.
      For delays tied to TopologySpec.seed, pass rng=np.random.default_rng(spec.seed).
    For edges with W[i,j]==0, delay is set to 0.
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be square (N,N).")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    dmin, dmax = float(delay_range_s[0]), float(delay_range_s[1])
    if dmin < 0 or dmax < dmin:
        raise ValueError("delay_range_s must satisfy 0 <= dmin <= dmax.")

    if rng is None:
        rng = np.random.default_rng(1)

    N = W.shape[0]
    D = np.zeros((N, N), dtype=int)

    idx = np.argwhere(W != 0.0)
    for i, j in idx:
        if (i == j) and (not include_self):
            continue
        delay_s = float(rng.uniform(dmin, dmax))
        D[i, j] = int(np.round(delay_s / dt))

    if not include_self:
        np.fill_diagonal(D, 0)
    return D


def spec_to_meta(spec: TopologySpec) -> Dict[str, Any]:
    meta = asdict(spec)
    # Keep floats as floats, not numpy types.
    meta["target_rho"] = None if spec.target_rho is None else float(spec.target_rho)
    meta["w_lognorm_mean"] = float(spec.w_lognorm_mean)
    meta["w_lognorm_sigma"] = float(spec.w_lognorm_sigma)
    return meta
