# src/nos.py
"""
Network-Optimised Spiking (NOS): canonical reference implementation.

Design goals:
- Single source of truth for NOS dynamics used across all scripts/notebooks.
- Deterministic when a RNG seed is provided.
- Supports: saturating excitability, recovery state, threshold jitter, smooth reset,
  and optional event-delayed coupling for network simulations.

"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Optional, Tuple, Union, Dict, Any, Iterable

import numpy as np


ArrayLike = Union[np.ndarray, float]


@dataclass(frozen=True)
class NOSParams:
    # Excitability
    alpha: float = 1.0
    kappa: float = 1.0  # kappa=0 gives pure quadratic alpha*v^2

    # Linear terms in v
    beta: float = 0.0
    gamma: float = 0.0
    lam: float = 0.0          # service / drain term on v
    chi: float = 0.0          # tether to v_rest
    v_rest: float = 0.0

    # Recovery dynamics
    a: float = 1.0
    b: float = 1.0
    mu: float = 0.0

    # Event generation
    theta: float = 1.0
    thresh_jitter_std: float = 0.0  # standard deviation added to theta per step
    rho_reset: float = 6.0          # smooth reset rate
    c_reset: float = 0.0            # reset centre c
    du_spike: float = 0.5           # recovery increment on spike

    # Numerical safety (optional)
    v_clip: Optional[float] = None  # e.g. 10.0 to avoid overflow in extreme runs

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NOSState:
    v: np.ndarray  # shape (N,)
    u: np.ndarray  # shape (N,)

    def copy(self) -> "NOSState":
        return NOSState(v=self.v.copy(), u=self.u.copy())


def f_sat(v: np.ndarray, alpha: float, kappa: float) -> np.ndarray:
    """
    Saturating excitability:
        f(v) = alpha * v^2 / (1 + kappa * v^2)
    Special case: kappa == 0 -> alpha * v^2
    """
    v2 = v * v
    if kappa == 0.0:
        return alpha * v2
    return (alpha * v2) / (1.0 + kappa * v2)


def f_sat_prime(v: np.ndarray, alpha: float, kappa: float) -> np.ndarray:
    """
    Derivative of f_sat with respect to v.
    """
    if kappa == 0.0:
        return 2.0 * alpha * v
    v2 = v * v
    denom = (1.0 + kappa * v2)
    # d/dv [ alpha v^2 / (1 + kappa v^2) ] = 2 alpha v / (1 + kappa v^2)^2
    return (2.0 * alpha * v) / (denom * denom)


def nos_drift(v: np.ndarray, u: np.ndarray, I: np.ndarray, p: NOSParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Continuous-time NOS drift (dv/dt, du/dt) without threshold/reset.

    dv = f_sat(v) + beta*v + gamma - u + I - lam*v - chi*(v - v_rest)
    du = a*(b*v - u) - mu*u
    """
    dv = (
        f_sat(v, p.alpha, p.kappa)
        + (p.beta - p.lam - p.chi) * v
        + (p.gamma + p.chi * p.v_rest)
        - u
        + I
    )
    du = p.a * (p.b * v - u) - p.mu * u
    return dv, du


def nos_step(
    state: NOSState,
    I: np.ndarray,
    dt: float,
    p: NOSParams,
    rng: Optional[np.random.Generator] = None,
    additive_noise: Optional[np.ndarray] = None,
) -> Tuple[NOSState, np.ndarray]:
    """
    One Euler step with thresholding and smooth reset.
    
    Differentiability note:
        The pullback reset (c_reset, rho_reset) is smooth in (v,u), but the spike event is
        a hard threshold (v_new >= theta). For end-to-end gradient training, use a surrogate
        gradient / soft spike indicator for the threshold in the training code.

    Returns:
        new_state, spikes (bool array, shape (N,))
    """
    if rng is None:
        rng = np.random.default_rng()

    v = state.v
    u = state.u

    if additive_noise is None:
        additive_noise = 0.0
    # Ensure vector
    if np.isscalar(additive_noise):
        additive_noise = np.full_like(v, float(additive_noise), dtype=float)

    dv, du = nos_drift(v, u, I, p)
    v_new = v + dt * dv + additive_noise
    u_new = u + dt * du

    if p.v_clip is not None:
        v_new = np.clip(v_new, -float(p.v_clip), float(p.v_clip))

    # Threshold with jitter (per neuron, per step)
    if p.thresh_jitter_std > 0.0:
        theta_eff = p.theta + rng.normal(0.0, p.thresh_jitter_std, size=v_new.shape)
    else:
        theta_eff = p.theta

    spikes = v_new >= theta_eff

    if np.any(spikes):
        # Smooth pullback reset (differentiable)
        # v <- c + (v - c) * exp(-rho_reset * dt)
        pull = np.exp(-p.rho_reset * dt)
        v_new[spikes] = p.c_reset + (v_new[spikes] - p.c_reset) * pull
        # Recovery kick
        u_new[spikes] = u_new[spikes] + p.du_spike

    return NOSState(v=v_new, u=u_new), spikes.astype(bool)


class EventDelayCoupling:
    """
    Event-delayed coupling: spikes from neuron j deliver weight to neuron i after delay_steps[i,j].

    The delivered input at time t is:
        I_syn[i,t] = sum_j (g * W[i,j]) * spike_j[t - delay(i,j)]

    Uses a circular buffer per target neuron i.

    Semantics (discrete time):
        Each call to step() advances an internal circular buffer by one discrete step.
        For an edge j→i with integer delay d = delay_steps[i,j]:
          - if spikes[j] is True on a call, the effective weight g*W[i,j] is scheduled
            for delivery d steps later;
          - d=0 means delivery on the same call to step().

        In simulate_nos_network(), spikes_prev (from the previous integration step) is
        passed into step(), so a zero-delay edge affects the next Euler update, which is
        the intended explicit-loop causality.

    Notes:
    - delays_steps must be nonnegative integers.
    - If delays_steps is None, delay is treated as 0 for all edges.
    """

    def __init__(
        self,
        W: np.ndarray,
        g: float = 1.0,
        delays_steps: Optional[np.ndarray] = None,
    ) -> None:
        W = np.asarray(W, dtype=float)
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("W must be a square (N,N) array.")
        self.W = W
        self.N = W.shape[0]
        self.g = float(g)

        if delays_steps is None:
            delays_steps = np.zeros_like(W, dtype=int)
        delays_steps = np.asarray(delays_steps, dtype=int)
        if delays_steps.shape != W.shape:
            raise ValueError("delays_steps must have the same shape as W.")
        if np.any(delays_steps < 0):
            raise ValueError("delays_steps must be nonnegative.")
        self.delays_steps = delays_steps

        self.max_delay = int(np.max(delays_steps))
        self._L = self.max_delay + 1
        self._buf = np.zeros((self.N, self._L), dtype=float)
        self._tidx = 0

        # Precompute outgoing adjacency lists for efficiency
        self._out: list[list[Tuple[int, float, int]]] = [[] for _ in range(self.N)]
        for i in range(self.N):
            for j in range(self.N):
                w = W[i, j]
                if w != 0.0:
                    d = int(delays_steps[i, j])
                    self._out[j].append((i, self.g * w, d))

    def step(self, spikes: np.ndarray) -> np.ndarray:
        """
        Advance coupling buffer by one step and enqueue new spike deliveries.

        Input:
            spikes: bool array shape (N,)

        Output:
            I_syn: float array shape (N,), delivered at current step
        """
        spikes = np.asarray(spikes, dtype=bool)
        if spikes.shape != (self.N,):
            raise ValueError("spikes must have shape (N,)")

        # Enqueue deliveries caused by provided spikes
        if np.any(spikes):
            for j in np.where(spikes)[0]:
                for (i, w_eff, d) in self._out[j]:
                    slot = (self._tidx + d) % self._L
                    self._buf[i, slot] += w_eff

        # Delivered now
        I_syn = self._buf[:, self._tidx].copy()
        # Clear the slot we just consumed
        self._buf[:, self._tidx] = 0.0

        # Advance time index
        self._tidx = (self._tidx + 1) % self._L
        return I_syn

    def reset(self) -> None:
        self._buf.fill(0.0)
        self._tidx = 0


def simulate_nos_network(
    T: float,
    dt: float,
    p: NOSParams,
    I_ext: Union[np.ndarray, Callable[[int, float], np.ndarray]],
    W: Optional[np.ndarray] = None,
    g: float = 1.0,
    delays_steps: Optional[np.ndarray] = None,
    v0: Optional[np.ndarray] = None,
    u0: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    noise_std: float = 0.0,
    record_spikes: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Simulate a NOS network (or a single uncoupled population if W is None).

    Args:
        T: total time (seconds or arbitrary units)
        dt: step size
        p: NOS parameters
        I_ext: either array of shape (steps, N) or a function f(k, t)->(N,)
        W: coupling matrix (N,N). If None, no coupling.
        g: coupling gain
        delays_steps: integer delays (N,N) in steps, optional
        v0, u0: initial states (N,)
        noise_std: Gaussian noise std added to v each step
        
        Noise scaling note:
            noise_std is per-step additive Gaussian noise applied to v. For a
            continuous-time diffusion term, scale the sampled noise by sqrt(dt) in the caller.
        record_spikes: whether to store spikes (steps, N)

    Returns dict with keys: t, v, u, spikes (optional)
    """
    if rng is None:
        rng = np.random.default_rng()

    steps = int(np.round(T / dt))
    if steps <= 0:
        raise ValueError("T/dt must give a positive number of steps.")

    # Infer N
    if W is not None:
        W = np.asarray(W, dtype=float)
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("W must be square (N,N).")
        N = W.shape[0]
        coupling = EventDelayCoupling(W=W, g=g, delays_steps=delays_steps)
    else:
        coupling = None
        # Try to infer N from I_ext
        if isinstance(I_ext, np.ndarray):
            if I_ext.ndim != 2:
                raise ValueError("I_ext array must have shape (steps, N).")
            N = I_ext.shape[1]
        else:
            raise ValueError("If W is None, provide I_ext as an array (steps, N) to infer N.")

    if v0 is None:
        v0 = np.zeros(N, dtype=float)
    if u0 is None:
        u0 = np.zeros(N, dtype=float)

    state = NOSState(v=np.asarray(v0, dtype=float).copy(), u=np.asarray(u0, dtype=float).copy())

    t = np.arange(steps) * dt
    V = np.zeros((steps, N), dtype=float)
    U = np.zeros((steps, N), dtype=float)
    S = np.zeros((steps, N), dtype=bool) if record_spikes else None

    spikes_prev = np.zeros(N, dtype=bool)

    for k in range(steps):
        tk = t[k]

        if isinstance(I_ext, np.ndarray):
            I_k = I_ext[k]
        else:
            I_k = I_ext(k, tk)

        I_k = np.asarray(I_k, dtype=float)
        if I_k.shape != (N,):
            raise ValueError("I_ext must produce shape (N,)")

        I_syn = coupling.step(spikes_prev) if coupling is not None else 0.0
        if np.isscalar(I_syn):
            I_syn = np.full(N, float(I_syn), dtype=float)

        noise = rng.normal(0.0, noise_std, size=N) if noise_std > 0.0 else 0.0
        state, spikes = nos_step(state=state, I=I_k + I_syn, dt=dt, p=p, rng=rng, additive_noise=noise)

        V[k, :] = state.v
        U[k, :] = state.u
        if record_spikes and S is not None:
            S[k, :] = spikes

        spikes_prev = spikes

    out: Dict[str, np.ndarray] = {"t": t, "v": V, "u": U}
    if record_spikes and S is not None:
        out["spikes"] = S.astype(np.uint8)
    return out
