# src/linearisation.py
"""
Linearisation utilities for NOS networks.

What this module is for
- Quick, reproducible spectral sanity checks used in the appendix (e.g., Gershgorin/eigenvalue scans).
- A no-delay, linearised 2N×2N block Jacobian for the continuous (v,u) drift.


Coupling model note
The Jacobians here treat coupling as a linear term for analysis convenience.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import warnings

from .nos import NOSParams, f_sat_prime
from .topology import spectral_radius


@dataclass(frozen=True)
class LinPoint:
    """
    Linearisation point for NOS node dynamics.
    """
    vstar: float = 0.0  # equilibrium / operating point for v
   
    coupling_through_nonlinearity: bool = False


def effective_dbar(p: NOSParams, lp: LinPoint) -> float:
    """
    Effective self-derivative term in dv/dt w.r.t v at the linearisation point.

    For the canonical drift:
      dv = f_sat(v) + (beta - lam - chi)*v + (gamma + chi*v_rest) - u + I

    so:
      ∂dv/∂v = f'(v*) + (beta - lam - chi)
    """
    v = np.array([lp.vstar], dtype=float)
    fp = float(f_sat_prime(v, p.alpha, p.kappa)[0])
    return fp + (p.beta - p.lam - p.chi)


def block_jacobian(
    W: np.ndarray,
    k: float,
    p: NOSParams,
    lp: LinPoint,
) -> np.ndarray:
    """
    Build the 2N×2N block Jacobian:

        [ A11   A12 ]
        [ A21   A22 ]

    with:
      A11 = dbar*I + k * W_eff
      A12 = -I
      A21 = (a*b)*I
      A22 = -(a+mu)*I

    If lp.coupling_through_nonlinearity=True then W_eff = f'(v*) * W,
    otherwise W_eff = W.
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be square (N,N).")
    N = W.shape[0]

    if lp.coupling_through_nonlinearity:
        warnings.warn(
            "LinPoint.coupling_through_nonlinearity is deprecated and non-canonical. "
            "NOS coupling is event-based (spike-driven additive input); this flag only applies a legacy "
            "Jacobian scaling by f'_sat(v*). Prefer leaving it False for paper-consistent analysis.",
            category=DeprecationWarning,
            stacklevel=2,
        )

    dbar = effective_dbar(p, lp)

    if lp.coupling_through_nonlinearity:
        v = np.array([lp.vstar], dtype=float)
        fp = float(f_sat_prime(v, p.alpha, p.kappa)[0])
        W_eff = fp * W
    else:
        W_eff = W

    A11 = dbar * np.eye(N) + float(k) * W_eff
    A12 = -np.eye(N)
    A21 = (p.a * p.b) * np.eye(N)
    A22 = -(p.a + p.mu) * np.eye(N)

    top = np.hstack([A11, A12])
    bot = np.hstack([A21, A22])
    return np.vstack([top, bot])


def leading_real_part_eig(J: np.ndarray) -> float:
    """
    Return max Re(eig(J)).
    """
    vals = np.linalg.eigvals(J)
    return float(np.max(np.real(vals)))


def is_stable_no_delay(W: np.ndarray, k: float, p: NOSParams, lp: LinPoint) -> bool:
    """
    Stability proxy for the no-delay Jacobian: max Re(eig(J)) < 0.
    """
    J = block_jacobian(W=W, k=k, p=p, lp=lp)
    return leading_real_part_eig(J) < 0.0


def find_k_star_bisect(
    W: np.ndarray,
    p: NOSParams,
    lp: LinPoint,
    k_lo: float = 0.0,
    k_hi: float = 50.0,
    max_expand: int = 30,
    tol: float = 1e-6,
    max_iter: int = 60,
) -> Tuple[float, Dict[str, Any]]:
    """
    Find critical k* where stability is lost (no-delay Jacobian) by bracketing + bisection.

    Returns:
      (k_star, info)

    info includes:
      - lead_lo, lead_hi: leading real-part eig at bracket ends
      - iters, bracket_expansions
    """
    W = np.asarray(W, dtype=float)

    def lead(k: float) -> float:
        J = block_jacobian(W=W, k=k, p=p, lp=lp)
        return leading_real_part_eig(J)

    klo = float(k_lo)
    khi = float(k_hi)
    lead_lo = lead(klo)
    lead_hi = lead(khi)

    # Expand upper bound until unstable (lead_hi > 0) or max_expand hit
    expansions = 0
    while lead_hi <= 0.0 and expansions < max_expand:
        khi *= 2.0
        lead_hi = lead(khi)
        expansions += 1

    if lead_lo > 0.0:
        raise ValueError("k_lo is already unstable; choose a smaller k_lo.")
    if lead_hi <= 0.0:
        raise ValueError("Failed to bracket instability: increase k_hi or max_expand.")

    # Bisection
    iters = 0
    while (khi - klo) > tol and iters < max_iter:
        km = 0.5 * (klo + khi)
        lm = lead(km)
        if lm <= 0.0:
            klo = km
            lead_lo = lm
        else:
            khi = km
            lead_hi = lm
        iters += 1

    k_star = 0.5 * (klo + khi)
    info = {
        "lead_lo": float(lead_lo),
        "lead_hi": float(lead_hi),
        "iters": int(iters),
        "bracket_expansions": int(expansions),
        "k_lo": float(k_lo),
        "k_hi_final": float(khi),
    }
    return float(k_star), info


def predict_k_star_hopf(
    W: np.ndarray,
    p: NOSParams,
    lp: LinPoint,
) -> Optional[float]:
    """
    Simple analytic predictor for Hopf-type crossing on the dominant mode.

    For a single eigenmode with eigenvalue lambda_W (real, W symmetric),
    the 2×2 mode Jacobian is:
        [ dbar + k*lambda_W   -1 ]
        [ a*b                -(a+mu) ]

    Hopf boundary when trace = 0:
        (dbar + k*lambda_W) - (a+mu) = 0
        => k = (a+mu - dbar)/lambda_W

    This Hopf crossing is admissible only if determinant > 0 at trace=0:
        a*b > (a+mu)^2

    We take lambda_W ≈ rho(W) for positive symmetric weights.

    Returns None if Hopf condition is not admissible or rho(W)=0.
    """
    dbar = effective_dbar(p, lp)
    if (p.a * p.b) <= (p.a + p.mu) ** 2:
        return None

    rho = spectral_radius(np.asarray(W, dtype=float))
    if rho <= 0.0:
        return None

    # If coupling is passed through f'(v*), absorb it into effective rho
    if lp.coupling_through_nonlinearity:
        v = np.array([lp.vstar], dtype=float)
        fp = float(f_sat_prime(v, p.alpha, p.kappa)[0])
        rho_eff = abs(fp) * rho
    else:
        rho_eff = rho

    if rho_eff <= 0.0:
        return None

    return float((p.a + p.mu - dbar) / rho_eff)


# ----------------------------
# Optional delay surrogate
# --------------------------

def block_jacobian_delay_surrogate(
    W: np.ndarray,
    delays_steps: np.ndarray,
    dt: float,
    k: float,
    p: NOSParams,
    lp: LinPoint,
    omega: float,
    attenuation_lam: float = 0.0,
) -> np.ndarray:
    """
    Build a complex-valued surrogate Jacobian that mimics phase/attenuation due to delays.

    This is NOT an exact DDE stability test. It mirrors the heuristic used in paper:
        W_eff[i,j](omega) = W[i,j] * exp( -(attenuation_lam + i*omega) * tau_ij )

    where tau_ij = delays_steps[i,j] * dt.
    """
    W = np.asarray(W, dtype=float)
    D = np.asarray(delays_steps, dtype=int)
    if W.shape != D.shape:
        raise ValueError("W and delays_steps must have the same shape.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if np.any(D < 0):
        raise ValueError("delays_steps must be nonnegative.")

    tau = D.astype(float) * float(dt)
    phase = np.exp(-(float(attenuation_lam) + 1j * float(omega)) * tau)

    if lp.coupling_through_nonlinearity:
        v = np.array([lp.vstar], dtype=float)
        fp = float(f_sat_prime(v, p.alpha, p.kappa)[0])
        W_eff = (fp * W) * phase
    else:
        W_eff = W * phase

    N = W.shape[0]
    dbar = effective_dbar(p, lp)

    A11 = dbar * np.eye(N, dtype=complex) + float(k) * W_eff.astype(complex)
    A12 = -np.eye(N, dtype=complex)
    A21 = (p.a * p.b) * np.eye(N, dtype=complex)
    A22 = -(p.a + p.mu) * np.eye(N, dtype=complex)

    top = np.hstack([A11, A12])
    bot = np.hstack([A21, A22])
    return np.vstack([top, bot])


def max_leading_real_part_over_omega(
    W: np.ndarray,
    delays_steps: np.ndarray,
    dt: float,
    k: float,
    p: NOSParams,
    lp: LinPoint,
    omega_grid: np.ndarray,
    attenuation_lam: float = 0.0,
) -> float:
    """
    Surrogate: max_{omega in grid} max Re(eig(J(omega))).
    """
    omega_grid = np.asarray(omega_grid, dtype=float)
    worst = -np.inf
    for om in omega_grid:
        J = block_jacobian_delay_surrogate(
            W=W,
            delays_steps=delays_steps,
            dt=dt,
            k=k,
            p=p,
            lp=lp,
            omega=float(om),
            attenuation_lam=attenuation_lam,
        )
        val = leading_real_part_eig(J)
        if val > worst:
            worst = val
    return float(worst)




def surrogate_effective_dbar(p: NOSParams, lp: LinPoint) -> float:
    """Preferred explicit name for `effective_dbar` (surrogate linearisation term)."""
    return effective_dbar(p=p, lp=lp)


def surrogate_block_jacobian_no_delay(
    W: np.ndarray,
    k: float,
    p: NOSParams,
    lp: LinPoint,
) -> np.ndarray:
    """
    Preferred explicit name for `block_jacobian`.

    This is a no-delay, continuous-drift linearisation used as a stability proxy.
    """
    return block_jacobian(W=W, k=k, p=p, lp=lp)


def surrogate_leading_real_part_eig(J: np.ndarray) -> float:
    """Preferred explicit name for `leading_real_part_eig`."""
    return leading_real_part_eig(J)


def surrogate_is_stable_no_delay(W: np.ndarray, k: float, p: NOSParams, lp: LinPoint) -> bool:
    """Preferred explicit name for `is_stable_no_delay`."""
    return is_stable_no_delay(W=W, k=k, p=p, lp=lp)


def surrogate_find_k_star_bisect_no_delay(
    W: np.ndarray,
    p: NOSParams,
    lp: LinPoint,
    k_lo: float = 0.0,
    k_hi: float = 50.0,
    max_expand: int = 30,
    tol: float = 1e-6,
    max_iter: int = 60,
) -> Tuple[float, Dict[str, Any]]:
    """Preferred explicit name for `find_k_star_bisect` (no-delay surrogate)."""
    return find_k_star_bisect(
        W=W,
        p=p,
        lp=lp,
        k_lo=k_lo,
        k_hi=k_hi,
        max_expand=max_expand,
        tol=tol,
        max_iter=max_iter,
    )


def surrogate_predict_k_star_hopf_mode(
    W: np.ndarray,
    p: NOSParams,
    lp: LinPoint,
) -> Optional[float]:
    """
    Preferred explicit name for `predict_k_star_hopf`.

    This is an analytic mode-level surrogate and relies on symmetry/real-mode assumptions.
    """
    return predict_k_star_hopf(W=W, p=p, lp=lp)


def surrogate_block_jacobian_delay_scan(
    W: np.ndarray,
    delays_steps: np.ndarray,
    dt: float,
    k: float,
    p: NOSParams,
    lp: LinPoint,
    omega: float,
    attenuation_lam: float = 0.0,
) -> np.ndarray:
    """Preferred explicit name for `block_jacobian_delay_surrogate`."""
    return block_jacobian_delay_surrogate(
        W=W,
        delays_steps=delays_steps,
        dt=dt,
        k=k,
        p=p,
        lp=lp,
        omega=omega,
        attenuation_lam=attenuation_lam,
    )


def surrogate_max_leading_real_part_over_omega(
    W: np.ndarray,
    delays_steps: np.ndarray,
    dt: float,
    k: float,
    p: NOSParams,
    lp: LinPoint,
    omega_grid: np.ndarray,
    attenuation_lam: float = 0.0,
) -> float:
    """Preferred explicit name for `max_leading_real_part_over_omega`."""
    return max_leading_real_part_over_omega(
        W=W,
        delays_steps=delays_steps,
        dt=dt,
        k=k,
        p=p,
        lp=lp,
        omega_grid=omega_grid,
        attenuation_lam=attenuation_lam,
    )
