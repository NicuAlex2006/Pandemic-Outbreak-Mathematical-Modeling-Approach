"""
Controlled SIRDQ epidemic — the single ground-truth dynamical system.

Compartments (fractions of one population, S + I + Q + R + D = 1):
    S  susceptible
    I  infectious & free      (drives transmission)
    Q  infectious & isolated  (no transmission, better care)
    R  recovered (immune; waning at rate omega, default 0)
    D  dead

Two controls, u = (u_L, u_Q) in [0, u_L_max] x [0, u_Q_max]:
    u_L  lockdown  -> cuts contacts:        beta(1 - u_L) S I
    u_Q  isolation -> rate I -> Q:          u_Q I

Dynamics (rates):
    dS = -beta(1 - u_L) S I              + omega R
    dI =  beta(1 - u_L) S I - (gamma + mu) I - u_Q I
    dQ =  u_Q I             - (gamma_Q + mu_Q) Q
    dR =  gamma I + gamma_Q Q            - omega R
    dD =  mu I + mu_Q Q

The five rates sum to zero -> mass is exactly conserved, which is what lets the
agent layer render the trajectory without drift.

Running cost (priced per unit time):
    L = w_I I + w_Q Q + V_D (mu I + mu_Q Q)
        + 0.5 c_L u_L**2 + 0.5 c_Q u_Q**2
        + 0.5 kappa max(0, I - I_cap)**2

The w_I I term (burden of *being infected*, not just dying) is what makes
suppression worthwhile independent of a small mu — it is the cure for the old
"optimal policy is to do nothing" collapse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Params:
    # --- epidemiology (rates per day) ---------------------------------------
    beta: float = 0.45        # transmission rate  (R0 ~ 3.6)
    gamma: float = 0.125      # recovery rate (free infectious)
    mu: float = 0.0013        # mortality rate (free infectious); ~1% CFR
    gamma_Q: float = 0.14     # recovery rate (isolated) — slightly better care
    mu_Q: float = 0.0005      # mortality rate (isolated) — better care
    omega: float = 0.0        # immunity waning (0 = single wave)

    # --- economics / cost weights -------------------------------------------
    V_D: float = 1.0          # value of a death (normalisation anchor)
    w_I: float = 0.05         # burden per unit time of a free infection
    w_Q: float = 0.03         # burden per unit time of an isolated case
    c_L: float = 0.5          # lockdown effort cost coeff (0.5 c_L u_L^2)
    c_Q: float = 0.3          # isolation effort cost coeff (0.5 c_Q u_Q^2)
    rho: float = 0.0          # discount rate (0 = finite horizon, lambda(T)=0)

    # --- hospital capacity penalty ------------------------------------------
    I_cap: float = 0.06       # "flatten under this line"
    kappa: float = 25.0       # penalty weight above capacity

    # --- control bounds ------------------------------------------------------
    # u_Q_max is *isolation capacity*: test-trace-isolate can only remove a
    # limited fraction of cases per day, so on its own it cannot hold R_eff < 1
    # for an R0 ~ 3.6 disease -> lockdown must supplement during the wave.
    u_L_max: float = 0.9      # essential activity never lets transmission -> 0
    u_Q_max: float = 0.08

    # --- horizon / discretisation -------------------------------------------
    T: float = 200.0          # days
    n_steps: int = 400        # dt = T / n_steps

    @property
    def dt(self) -> float:
        return self.T / self.n_steps


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

def deriv(state, u_L, u_Q, p: Params):
    """Return (dS, dI, dQ, dR, dD) for one state and controls."""
    S, I, Q, R, D = state
    infection = p.beta * (1.0 - u_L) * S * I
    dS = -infection + p.omega * R
    dI = infection - (p.gamma + p.mu) * I - u_Q * I
    dQ = u_Q * I - (p.gamma_Q + p.mu_Q) * Q
    dR = p.gamma * I + p.gamma_Q * Q - p.omega * R
    dD = p.mu * I + p.mu_Q * Q
    return np.array([dS, dI, dQ, dR, dD])


def step_heun(state, u_L, u_Q, u_L_next, u_Q_next, p: Params):
    """One Heun (improved-Euler) step. Controls given at this and next node."""
    dt = p.dt
    k1 = deriv(state, u_L, u_Q, p)
    pred = state + dt * k1
    k2 = deriv(pred, u_L_next, u_Q_next, p)
    return state + 0.5 * dt * (k1 + k2)


def integrate(x0, u_L, u_Q, p: Params):
    """Forward-integrate the controlled ODE with Heun (RK2) over the grid.

    u_L, u_Q are arrays of length n_steps+1 (value at each time node).
    Returns X with shape (n_steps+1, 5): columns S, I, Q, R, D.

    Heun is the default inside the PMP sweep: it is O(dt^2) accurate yet cheap,
    which keeps every forward pass of the forward-backward iteration fast. Use
    `integrate_rk4` for the single high-fidelity trajectory that is plotted and
    animated once the controls have converged.
    """
    n = p.n_steps
    X = np.empty((n + 1, 5))
    X[0] = x0
    for k in range(n):
        X[k + 1] = step_heun(X[k], u_L[k], u_Q[k], u_L[k + 1], u_Q[k + 1], p)
        # numerical hygiene: keep fractions in [0, 1]
        np.clip(X[k + 1], 0.0, 1.0, out=X[k + 1])
    return X


def integrate_rk4(x0, u_L, u_Q, p: Params):
    """Forward-integrate with classic RK4 (O(dt^4)).

    The controls are piecewise data on the grid, so the two RK4 stage
    evaluations at the midpoint use the linear interpolant
    u_mid = (u[k] + u[k+1]) / 2 — consistent with the trapezoidal control
    profile the PMP sweep converges to. Same signature/shape as `integrate`.
    """
    n = p.n_steps
    dt = p.dt
    X = np.empty((n + 1, 5))
    X[0] = x0
    for k in range(n):
        uL0, uQ0 = u_L[k], u_Q[k]
        uL1, uQ1 = u_L[k + 1], u_Q[k + 1]
        uLm, uQm = 0.5 * (uL0 + uL1), 0.5 * (uQ0 + uQ1)
        x = X[k]
        k1 = deriv(x, uL0, uQ0, p)
        k2 = deriv(x + 0.5 * dt * k1, uLm, uQm, p)
        k3 = deriv(x + 0.5 * dt * k2, uLm, uQm, p)
        k4 = deriv(x + dt * k3, uL1, uQ1, p)
        X[k + 1] = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        np.clip(X[k + 1], 0.0, 1.0, out=X[k + 1])
    return X


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

def r_eff(S, u_L, u_Q, p: Params):
    """Effective reproduction number under control."""
    denom = p.gamma + p.mu + u_Q
    return p.beta * (1.0 - u_L) * S / np.where(denom > 0, denom, 1e-9)


def r0(p: Params) -> float:
    """Basic reproduction number (no control, S=1)."""
    return p.beta / (p.gamma + p.mu)
