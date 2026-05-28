"""
Hamilton-Jacobi-Bellman solver for optimal control of the SIR epidemic model.

Cost model (single user-facing knob: care in [0, 1]):

    L(I, u; care) = care * I * (1 + KAPPA * I)
                   + (1 - care) * C_U_BASE * u^2

  - care = 0  : only the control cost matters -> u* = 0 (do nothing)
  - care = 1  : only the (convex) infection cost matters -> u* saturates
  - care = 0.5: balanced interior optimum
  - KAPPA     : convexity in I, implicit hospital-overflow signal
                (no separate I_cap / w_h parameters required)

Subject to:
    dS/dt = -beta * (1 - u) * S * I
    dI/dt =  beta * (1 - u) * S * I - gamma * I - mu * I
    dR/dt =  gamma * I
    u in [0, 1]

Includes an optimal stopping extension (variational inequality): the
controller may permanently drop lockdown at time tau, accepting the
uncontrolled "do nothing" cost g(s, i).

Solved via backward induction on a 2-D (S, I) grid with upwind finite
differences, Lax-Friedrichs viscosity, and explicit Euler time-stepping.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------

BETA = 0.4
GAMMA = 0.1
MU = 0.005           # mortality rate
CARE = 0.5           # default infection-care weight (0=ignore, 1=zero-tolerance)
KAPPA = 10.0         # convexity of I-cost (implicit hospital-overflow)
C_U_BASE = 5.0       # baseline control-cost scale (auto-tuned via care)
T = 10.0             # time horizon
NS = 51              # grid points in S direction
NI = 51              # grid points in I direction
CFL_SAFETY = 0.4     # fraction of CFL limit
LF_VISCOSITY = 0.45  # Lax-Friedrichs viscosity (must exceed max wave speed)

CARE_EPS = 0.02      # clamp care into [CARE_EPS, 1 - CARE_EPS] for stability


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def build_grid(nS, nI):
    s = np.linspace(0, 1, nS)
    i = np.linspace(0, 1, nI)
    dS = s[1] - s[0]
    dI = i[1] - i[0]
    S_grid, I_grid = np.meshgrid(s, i, indexing="ij")
    return s, i, dS, dI, S_grid, I_grid


# ---------------------------------------------------------------------------
# Upwind spatial gradients (vectorised)
# ---------------------------------------------------------------------------

def upwind_gradients(V, dS, dI):
    # Forward difference in S (drift_S <= 0 always => upwind = forward)
    dVdS = np.zeros_like(V)
    dVdS[:-1, :] = (V[1:, :] - V[:-1, :]) / dS
    dVdS[-1, :] = dVdS[-2, :]

    # Backward difference in I (for drift_I > 0 regions)
    dVdI_bwd = np.zeros_like(V)
    dVdI_bwd[:, 1:] = (V[:, 1:] - V[:, :-1]) / dI

    # Forward difference in I (for drift_I < 0 regions)
    dVdI_fwd = np.zeros_like(V)
    dVdI_fwd[:, :-1] = (V[:, 1:] - V[:, :-1]) / dI

    return dVdS, dVdI_bwd, dVdI_fwd


# ---------------------------------------------------------------------------
# Optimal control and Hamiltonian (vectorised)
# ---------------------------------------------------------------------------

def _clamp_care(care):
    return float(np.clip(care, CARE_EPS, 1.0 - CARE_EPS))


def _effective_cu(care):
    """Control-cost coefficient = (1 - care) * C_U_BASE.

    Why: cost  L = care * I*(1+kappa I) + (1-care) * c_u * u^2.
    At care -> 0  control cost dominates  -> u* -> 0.
    At care -> 1  control cost vanishes   -> u* -> 1.
    """
    c = _clamp_care(care)
    return (1.0 - c) * C_U_BASE


BARRIER_EPS = 1e-3  # log-barrier weight (small enough to not distort CFL)


def optimal_control(S_grid, I_grid, dVdS, dVdI, beta, c_u):
    """Optimal control with smooth soft-clamp barrier (no hard clip).

    Derivation: stationary condition  d/du [c_u/2 u^2 + (dVdI - dVdS) * (-beta u S I)] = 0
        =>  u_raw = beta * S * I * (dVdI - dVdS) / c_u
    Map u_raw through u* = max(u_raw, 0) / (1 + max(u_raw, 0))
    which is C0 (C1 except at 0 where gradient is continuous from right),
    maps [0, inf) -> [0, 1), and u_raw=0 -> 0, u_raw=inf -> 1.
    """
    grad_diff = np.clip(dVdI - dVdS, -1e6, 1e6)
    c_u_safe = max(float(c_u), 1e-3)
    u_raw = beta * S_grid * I_grid * grad_diff / c_u_safe
    u_pos = np.maximum(u_raw, 0.0)
    u_star = u_pos / (1.0 + u_pos)
    return u_star


def hamiltonian(S_grid, I_grid, dVdS, dVdI_bwd, dVdI_fwd, u,
                care, beta, gamma, mu=0.0, c_u=None):
    if c_u is None:
        c_u = _effective_cu(care)
    care_c = _clamp_care(care)

    infection = beta * (1.0 - u) * S_grid * I_grid
    drift_S = -infection
    drift_I = infection - gamma * I_grid - mu * I_grid

    drift_I_pos = np.maximum(drift_I, 0.0)
    drift_I_neg = np.minimum(drift_I, 0.0)

    # Log-barrier: keeps u away from {0, 1} so optimal_control stays smooth.
    _eps2 = 1e-8
    barrier = -BARRIER_EPS * (np.log(u + _eps2) + np.log(1.0 - u + _eps2))

    # New cost: care-weighted convex infection term + control penalty
    infection_cost = care_c * I_grid * (1.0 + KAPPA * I_grid)
    control_cost = (c_u / 2.0) * u**2

    running_cost = infection_cost + control_cost + barrier
    return (running_cost
            + dVdS * drift_S
            + dVdI_bwd * drift_I_pos
            + dVdI_fwd * drift_I_neg)


# ---------------------------------------------------------------------------
# Do-nothing (uncontrolled) cost g(s, i) — vectorised forward Euler
# ---------------------------------------------------------------------------

def compute_do_nothing_cost(S_grid, I_grid, beta, gamma, care, T,
                            mu=0.0, n_fwd=2000):
    care_c = _clamp_care(care)
    dt_fwd = T / n_fwd
    S_fwd = S_grid.copy()
    I_fwd = I_grid.copy()
    g_array = np.zeros_like(S_grid)

    for _ in range(n_fwd):
        infection = beta * S_fwd * I_fwd
        dS = -infection
        dI = infection - gamma * I_fwd - mu * I_fwd
        g_array += care_c * I_fwd * (1.0 + KAPPA * I_fwd) * dt_fwd
        S_fwd = np.clip(S_fwd + dS * dt_fwd, 0.0, 1.0)
        I_fwd = np.clip(I_fwd + dI * dt_fwd, 0.0, 1.0)

    return g_array


# ---------------------------------------------------------------------------
# Backward induction with optimal stopping (variational inequality)
# ---------------------------------------------------------------------------

def solve_hjb(beta=BETA, gamma=GAMMA, care=CARE, T=T, nS=NS, nI=NI,
              mu=MU, c_u=None):
    care_c = _clamp_care(care)
    if c_u is None:
        c_u = _effective_cu(care_c)
    s, i, dS, dI, S_grid, I_grid = build_grid(nS, nI)

    max_speed = beta + gamma + mu
    dt = CFL_SAFETY * min(dS, dI) / max_speed
    n_steps = int(np.ceil(T / dt))
    dt = T / n_steps

    nu_S = LF_VISCOSITY * dS * max_speed
    nu_I = LF_VISCOSITY * dI * max_speed

    print(f"  care = {care_c:.3f}  c_u = {c_u:.4f}")
    print(f"  dt = {dt:.6f}, n_steps = {n_steps}")

    print("  Computing do-nothing cost g(s, i)...")
    g_array = compute_do_nothing_cost(S_grid, I_grid, beta, gamma,
                                      care_c, T, mu=mu)
    g_max = float(g_array.max())
    print(f"  g range: [{g_array.min():.4f}, {g_max:.4f}]")

    # Terminal value V(T, s, i) = 0 (no terminal penalty); start from a
    # plausible tail-cost surface to speed convergence.
    tail_rate = gamma + mu
    V = care_c * (1.0 + KAPPA * I_grid) * I_grid / max(tail_rate, 1e-6)
    t_remaining = T

    snap_steps = {
        0: "T",
        n_steps // 2: "T/2",
        n_steps - 1: "0",
    }
    stopping_snapshots = {}

    for step in range(n_steps):
        dVdS, dVdI_bwd, dVdI_fwd = upwind_gradients(V, dS, dI)
        dVdI_central = 0.5 * (dVdI_bwd + dVdI_fwd)
        u = optimal_control(S_grid, I_grid, dVdS, dVdI_central, beta, c_u=c_u)
        H = hamiltonian(S_grid, I_grid, dVdS, dVdI_bwd, dVdI_fwd, u,
                        care_c, beta, gamma, mu=mu, c_u=c_u)

        lap = np.zeros_like(V)
        lap[1:-1, :] += nu_S * (V[2:, :] - 2*V[1:-1, :] + V[:-2, :]) / dS**2
        lap[:, 1:-1] += nu_I * (V[:, 2:] - 2*V[:, 1:-1] + V[:, :-2]) / dI**2

        V = V + dt * (H + lap)

        np.clip(V, 0.0, g_max, out=V)
        np.nan_to_num(V, copy=False, nan=0.0, posinf=g_max, neginf=0.0)

        V = np.minimum(V, g_array)

        t_remaining -= dt

        if step in snap_steps:
            label = snap_steps[step]
            mask = np.isclose(V, g_array, rtol=1e-4, atol=1e-6)
            stopping_snapshots[label] = mask.copy()

        if step % 500 == 0:
            stopped_frac = np.mean(np.isclose(V, g_array, rtol=1e-4, atol=1e-6))
            print(f"  t = {t_remaining:.3f}  V_max = {V.max():.4f}  "
                  f"stopped = {stopped_frac:.1%}")

    dVdS, dVdI_bwd, dVdI_fwd = upwind_gradients(V, dS, dI)
    dVdI_central = 0.5 * (dVdI_bwd + dVdI_fwd)
    u_opt = optimal_control(S_grid, I_grid, dVdS, dVdI_central, beta, c_u=c_u)

    stopped_frac = float(np.mean(np.isclose(V, g_array, rtol=1e-4, atol=1e-6)))

    return s, i, V, u_opt, g_array, stopping_snapshots, stopped_frac


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_results(s, i, V, u_opt, stopping_snapshots):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    c = ax.contourf(i, s, V, levels=40, cmap="viridis")
    fig.colorbar(c, ax=ax, label="V(0, S, I)")
    ax.set_xlabel("I (infected)")
    ax.set_ylabel("S (susceptible)")
    ax.set_title("Value function at t = 0")

    ax = axes[1]
    c = ax.contourf(i, s, u_opt, levels=40, cmap="coolwarm")
    fig.colorbar(c, ax=ax, label="u*(0, S, I)")
    ax.set_xlabel("I (infected)")
    ax.set_ylabel("S (susceptible)")
    ax.set_title("Optimal control at t = 0")

    plt.tight_layout()

    time_labels = ["T", "T/2", "0"]
    available = [t for t in time_labels if t in stopping_snapshots]
    if available:
        fig2, axes2 = plt.subplots(1, len(available), figsize=(5 * len(available), 4))
        if len(available) == 1:
            axes2 = [axes2]
        for ax, t_label in zip(axes2, available):
            mask = stopping_snapshots[t_label].astype(float)
            ax.contourf(i, s, mask, levels=[0, 0.5, 1], colors=["#2166ac", "#b2182b"])
            ax.set_xlabel("I (infected)")
            ax.set_ylabel("S (susceptible)")
            ax.set_title(f"Stopping region at t = {t_label}")
            ax.text(0.02, 0.95, "■ stop (do nothing)\n■ continue (lockdown)",
                    transform=ax.transAxes, va="top", fontsize=8,
                    color="white", family="monospace")
        fig2.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"HJB solver: T={T}, grid={NS}x{NI}")
    print(f"  beta={BETA}, gamma={GAMMA}, mu={MU}, care={CARE}, "
          f"kappa={KAPPA}, c_u_base={C_U_BASE}\n")

    s, i, V, u_opt, g_array, stopping_snapshots, stopped_frac = solve_hjb()

    print(f"\nDone. V range: [{V.min():.4f}, {V.max():.4f}]")
    print(f"g range: [{g_array.min():.4f}, {g_array.max():.4f}]")
    print(f"u* range: [{u_opt.min():.4f}, {u_opt.max():.4f}]")
    print(f"Stopping region at t=0: {stopped_frac:.1%} of grid")

    fig = plot_results(s, i, V, u_opt, stopping_snapshots)
    plt.show()
