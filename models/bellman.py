"""
Vectorised finite-horizon Dynamic-Programming (Bellman) solver for the
quarantine-controlled epidemic, on the 2-D state space y = (I, Q).

Mathematical ground truth
--------------------------
State          y = (I, Q)                      I = infected, Q = quarantined
Control        u in [0, u_max]                 quarantine intensity
Dynamics (rates, explicit-Euler integrated with step dt):
    dI/dt = beta * (1 - I - Q) * I - (gamma + mu) * I - u * I
    dQ/dt = u * I            - (gamma_Q + mu_Q) * Q
    I_next = I + dt * dI/dt ,   Q_next = Q + dt * dQ/dt
Running cost   L(I, Q, u) = C_u * u**2 + W * (mu * I + mu_Q * Q)
Terminal cost  g(I, Q)    = W * (I + Q)
Bellman        V_k(I, Q)  = min_u [ L(I, Q, u) * dt + V_{k+1}(I_next, Q_next) ]

Design notes (HPC)
------------------
* The spatial loops over (I, Q) and the control loop over u are eliminated:
  every quantity is broadcast into a single (NI, NQ, NU) tensor and reduced
  with `min`/`argmin` over the control axis.
* The backward recursion over time *cannot* be vectorised -- V_k depends on
  V_{k+1} -- so exactly one explicit time loop remains. Because the system is
  autonomous (dynamics independent of k), the continuous->grid index map and
  the running-cost tensor are computed ONCE, outside that loop.
* The continuous next-state is mapped back to grid indices with pure array
  math (nearest grid node via round + clip + integer cast), replacing any
  scalar `get_index` helper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
IntArray = NDArray[np.intp]


# ---------------------------------------------------------------------------
# Infrastructure: physics params vs. grid/simulation spec (separated)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EpiParams:
    """Pure physics / economics of the (I, Q) controlled epidemic."""
    beta: float = 0.4        # transmission rate
    gamma: float = 0.1       # recovery rate (infected)
    mu: float = 0.005        # mortality rate (infected)
    gamma_Q: float = 0.12    # recovery rate (quarantined)
    mu_Q: float = 0.002      # mortality rate (quarantined)
    C_u: float = 5.0e3       # control-effort cost weight  ($ per u**2)
    W: float = 1.0e6         # welfare / death-cost weight ($ per unit mass)
    u_max: float = 1.0       # maximum quarantine intensity


@dataclass(frozen=True)
class GridSpec:
    """Discretisation of the state space, control set and time horizon."""
    NI: int = 81             # grid nodes along I
    NQ: int = 81             # grid nodes along Q
    NU: int = 41             # control samples in [0, u_max]
    I_max: float = 1.0
    Q_max: float = 1.0
    T: float = 10.0          # horizon (days)
    N_steps: int = 200       # backward time steps

    @property
    def dt(self) -> float:
        return self.T / self.N_steps


@dataclass
class BellmanResult:
    """Solver output."""
    I_axis: Array                    # (NI,)
    Q_axis: Array                    # (NQ,)
    u_axis: Array                    # (NU,)
    V0: Array                        # (NI, NQ)  value at t = 0
    policy0: Array                   # (NI, NQ)  optimal u at t = 0
    policy_stack: Array = field(repr=False)  # (N_steps, NI, NQ) optimal u over time


# ---------------------------------------------------------------------------
# Grid construction (np.meshgrid)
# ---------------------------------------------------------------------------

def build_state_grid(
    grid: GridSpec,
    u_max: float,
) -> Tuple[Array, Array, Array, Array, Array]:
    """Return (I_axis, Q_axis, u_axis, I_grid, Q_grid) with 'ij' indexing."""
    I_axis = np.linspace(0.0, grid.I_max, grid.NI)
    Q_axis = np.linspace(0.0, grid.Q_max, grid.NQ)
    u_axis = np.linspace(0.0, u_max, grid.NU)
    I_grid, Q_grid = np.meshgrid(I_axis, Q_axis, indexing="ij")  # (NI, NQ)
    return I_axis, Q_axis, u_axis, I_grid, Q_grid


# ---------------------------------------------------------------------------
# Core solver -- fully vectorised backward induction
# ---------------------------------------------------------------------------

def solve_bellman_backwards(
    params: EpiParams = EpiParams(),
    grid: GridSpec = GridSpec(),
    store_policy_stack: bool = True,
) -> BellmanResult:
    """Finite-horizon DP via vectorised backward induction.

    All (I, Q, u) sweeps are broadcast into rank-3 tensors; the only Python
    loop is the irreducible backward recursion over time.
    """
    I_axis, Q_axis, u_axis, I_grid, Q_grid = build_state_grid(grid, params.u_max)
    dt = grid.dt

    # --- broadcast state (NI, NQ, 1) against control (1, 1, NU) -------------
    I3 = I_grid[:, :, None]                       # (NI, NQ, 1)
    Q3 = Q_grid[:, :, None]                       # (NI, NQ, 1)
    u3 = u_axis[None, None, :]                     # (1, 1, NU)

    S3 = 1.0 - I3 - Q3                             # susceptible mass = 1 - I - Q
    dI = params.beta * S3 * I3 - (params.gamma + params.mu) * I3 - u3 * I3
    dQ = u3 * I3 - (params.gamma_Q + params.mu_Q) * Q3

    I_next = np.clip(I3 + dt * dI, 0.0, grid.I_max)   # (NI, NQ, NU)
    Q_next = np.clip(Q3 + dt * dQ, 0.0, grid.Q_max)   # (NI, NQ, NU)

    # --- vectorised continuous -> grid bilinear weights (replaces get_index) -
    # Nearest-neighbour snapping makes V blocky and hides the C_u*u**2 term;
    # bilinear interpolation gives a smooth Bellman operator so the policy
    # responds continuously to the cost. All weights/indices are constant in
    # time (autonomous dynamics) -> computed once, outside the loop.
    dI_node = I_axis[1] - I_axis[0]
    dQ_node = Q_axis[1] - Q_axis[0]
    fi = I_next / dI_node                              # (NI, NQ, NU) frac coords
    fj = Q_next / dQ_node
    i0 = np.clip(np.floor(fi), 0, grid.NI - 1).astype(np.intp)
    j0 = np.clip(np.floor(fj), 0, grid.NQ - 1).astype(np.intp)
    i1 = np.clip(i0 + 1, 0, grid.NI - 1)
    j1 = np.clip(j0 + 1, 0, grid.NQ - 1)
    wi = np.clip(fi - i0, 0.0, 1.0)                    # weight toward i1
    wj = np.clip(fj - j0, 0.0, 1.0)
    f00 = i0 * grid.NQ + j0                            # flat indices of 4 corners
    f01 = i0 * grid.NQ + j1
    f10 = i1 * grid.NQ + j0
    f11 = i1 * grid.NQ + j1
    w00 = (1.0 - wi) * (1.0 - wj)                      # bilinear weights
    w01 = (1.0 - wi) * wj
    w10 = wi * (1.0 - wj)
    w11 = wi * wj

    # --- running-cost tensor L(I, Q, u), constant in time ------------------
    L = params.C_u * u3**2 + params.W * (params.mu * I3 + params.mu_Q * Q3)
    Ldt = L * dt                                       # (NI, NQ, NU)

    # --- terminal condition V_N = g(I, Q) ----------------------------------
    V_next = params.W * (I_grid + Q_grid)              # (NI, NQ)

    policy_stack = (
        np.empty((grid.N_steps, grid.NI, grid.NQ), dtype=np.float64)
        if store_policy_stack else None
    )

    # --- backward recursion (the only loop; over TIME only) ----------------
    policy0: Array | None = None
    for k in range(grid.N_steps - 1, -1, -1):
        # bilinear gather of V_{k+1} at every (I_next, Q_next): (NI, NQ, NU)
        Vr = V_next.ravel()
        V_interp = (w00 * Vr[f00] + w01 * Vr[f01]
                    + w10 * Vr[f10] + w11 * Vr[f11])
        candidates = Ldt + V_interp                    # (NI, NQ, NU)
        best = candidates.argmin(axis=2)               # (NI, NQ)
        V_next = np.take_along_axis(
            candidates, best[:, :, None], axis=2
        )[:, :, 0]                                      # (NI, NQ) == min over u
        policy0 = u_axis[best]                          # (NI, NQ)
        if store_policy_stack:
            policy_stack[k] = policy0

    return BellmanResult(
        I_axis=I_axis, Q_axis=Q_axis, u_axis=u_axis,
        V0=V_next, policy0=policy0,
        policy_stack=policy_stack if store_policy_stack else np.empty((0,)),
    )


# ---------------------------------------------------------------------------
# QA reference: naive triple-nested-loop solver (ground-truth oracle)
# ---------------------------------------------------------------------------

def _bilinear(V: Array, I_nx: float, Q_nx: float,
              dI: float, dQ: float, NI: int, NQ: int) -> float:
    """Scalar bilinear lookup -- the helper the vectorised path replaces."""
    fi, fj = I_nx / dI, Q_nx / dQ
    i0 = min(max(int(np.floor(fi)), 0), NI - 1)
    j0 = min(max(int(np.floor(fj)), 0), NQ - 1)
    i1 = min(i0 + 1, NI - 1)
    j1 = min(j0 + 1, NQ - 1)
    wi = min(max(fi - i0, 0.0), 1.0)
    wj = min(max(fj - j0, 0.0), 1.0)
    return ((1.0 - wi) * (1.0 - wj) * V[i0, j0]
            + (1.0 - wi) * wj * V[i0, j1]
            + wi * (1.0 - wj) * V[i1, j0]
            + wi * wj * V[i1, j1])


def solve_bellman_naive(
    params: EpiParams, grid: GridSpec
) -> Array:
    """Explicit nested-loop oracle. Slow, obviously correct -- QA use only."""
    I_axis = np.linspace(0.0, grid.I_max, grid.NI)
    Q_axis = np.linspace(0.0, grid.Q_max, grid.NQ)
    u_axis = np.linspace(0.0, params.u_max, grid.NU)
    dt = grid.dt

    V_next = np.empty((grid.NI, grid.NQ))
    for a in range(grid.NI):
        for b in range(grid.NQ):
            V_next[a, b] = params.W * (I_axis[a] + Q_axis[b])  # terminal g

    for _k in range(grid.N_steps - 1, -1, -1):
        V_cur = np.empty_like(V_next)
        for a in range(grid.NI):
            I = I_axis[a]
            for b in range(grid.NQ):
                Q = Q_axis[b]
                S = 1.0 - I - Q
                best = np.inf
                for c in range(grid.NU):
                    u = u_axis[c]
                    dI = params.beta * S * I - (params.gamma + params.mu) * I - u * I
                    dQ = u * I - (params.gamma_Q + params.mu_Q) * Q
                    I_nx = min(max(I + dt * dI, 0.0), grid.I_max)
                    Q_nx = min(max(Q + dt * dQ, 0.0), grid.Q_max)
                    dI_node = I_axis[1] - I_axis[0]
                    dQ_node = Q_axis[1] - Q_axis[0]
                    v_nx = _bilinear(V_next, I_nx, Q_nx, dI_node, dQ_node,
                                     grid.NI, grid.NQ)
                    L = params.C_u * u**2 + params.W * (params.mu * I + params.mu_Q * Q)
                    cand = L * dt + v_nx
                    if cand < best:
                        best = cand
                V_cur[a, b] = best
        V_next = V_cur
    return V_next


def verify_against_naive(
    params: EpiParams | None = None,
    grid: GridSpec | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-6,
) -> bool:
    """QA gate: assert the vectorised value function equals the naive oracle."""
    params = params or EpiParams()
    # small grid so the O(N_steps * NI * NQ * NU) oracle is tractable
    grid = grid or GridSpec(NI=21, NQ=21, NU=11, N_steps=40)
    fast = solve_bellman_backwards(params, grid, store_policy_stack=False).V0
    slow = solve_bellman_naive(params, grid)
    max_abs = float(np.max(np.abs(fast - slow)))
    ok = np.allclose(fast, slow, rtol=rtol, atol=atol)
    print(f"[QA] grid={grid.NI}x{grid.NQ}x{grid.NU} steps={grid.N_steps}  "
          f"max|ΔV| = {max_abs:.3e}  ->  {'MATCH' if ok else 'MISMATCH'}")
    return bool(ok)


if __name__ == "__main__":
    import time

    print("Vectorised (I, Q) finite-horizon Bellman solver\n")

    # 1) QA verification against the nested-loop oracle.
    assert verify_against_naive(), "vectorised solver disagrees with oracle!"

    # 2) Full-resolution timing.
    p, g = EpiParams(), GridSpec()
    t0 = time.perf_counter()
    res = solve_bellman_backwards(p, g)
    dt_ms = (time.perf_counter() - t0) * 1e3
    print(f"\nFull solve  grid={g.NI}x{g.NQ}x{g.NU}  steps={g.N_steps}  "
          f"in {dt_ms:.1f} ms")
    print(f"V0  range: [${res.V0.min():,.0f}, ${res.V0.max():,.0f}]")
    print(f"u*0 range: [{res.policy0.min():.3f}, {res.policy0.max():.3f}]")
