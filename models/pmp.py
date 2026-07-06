"""
Pontryagin (PMP) optimal control of the SIRDQ epidemic via a forward-backward
sweep — the mathematical backbone the whole app is built on.

State (dynamically relevant when omega = 0):  x = (S, I, Q)
Costates:                                       lambda = (lS, lI, lQ)
Deaths/recovered are priced through the running cost, so R and D do not feed
back and are reconstructed afterwards from conservation.

Current-value Hamiltonian (rho-discounted):
    H = L + lS dS + lI dI + lQ dQ
with
    L = w_I I + w_Q Q + V_D(mu I + mu_Q Q)
        + 0.5 c_L u_L^2 + 0.5 c_Q u_Q^2 + 0.5 kappa h^2 ,   h = max(0, I - I_cap)

Optimal controls (interior, then clipped) — each control sits in exactly one
term, so the minimisation is closed-form and decoupled:
    u_L* = clip( beta S I (lI - lS) / c_L , 0, u_L_max )
    u_Q* = clip(        I (lI - lQ) / c_Q , 0, u_Q_max )

Costate ODEs (current value, lambda_dot = rho lambda - dH/dx):
    lS' = rho lS - A I (lI - lS)
    lI' = rho lI - [ w_I + V_D mu + kappa h - A S lS
                     + lI (A S - (gamma + mu + u_Q)) + lQ u_Q ]
    lQ' = rho lQ - [ w_Q + V_D mu_Q - lQ (gamma_Q + mu_Q) ]
with A = beta (1 - u_L) and transversality lambda(T) = 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.sirdq import Params, integrate, integrate_rk4, r_eff, r0


@dataclass
class Trajectory:
    t: np.ndarray
    S: np.ndarray
    I: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    D: np.ndarray
    u_L: np.ndarray
    u_Q: np.ndarray
    Reff: np.ndarray
    params: Params
    iters: int = 0
    converged: bool = False

    def at(self, time: float) -> dict:
        """Linear sample of the trajectory at a given day."""
        time = float(np.clip(time, self.t[0], self.t[-1]))
        S = float(np.interp(time, self.t, self.S))
        I = float(np.interp(time, self.t, self.I))
        Q = float(np.interp(time, self.t, self.Q))
        R = float(np.interp(time, self.t, self.R))
        D = float(np.interp(time, self.t, self.D))
        return {
            "S": S, "I": I, "Q": Q, "R": R, "D": D,
            "u_L": float(np.interp(time, self.t, self.u_L)),
            "u_Q": float(np.interp(time, self.t, self.u_Q)),
            "Reff": float(np.interp(time, self.t, self.Reff)),
        }


def solve(p: Params, x0=None, max_iter: int = 600, tol: float = 1e-6,
          damp: float = 0.1) -> Trajectory:
    """Forward-backward sweep for the optimal (u_L, u_Q) trajectory."""
    n = p.n_steps
    dt = p.dt
    t = np.linspace(0.0, p.T, n + 1)

    if x0 is None:
        I0 = 0.001
        x0 = np.array([1.0 - I0, I0, 0.0, 0.0, 0.0])
    x0 = np.asarray(x0, dtype=float)

    u_L = np.zeros(n + 1)
    u_Q = np.zeros(n + 1)

    lS = np.zeros(n + 1)
    lI = np.zeros(n + 1)
    lQ = np.zeros(n + 1)

    X = integrate(x0, u_L, u_Q, p)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        # --- forward state ---------------------------------------------------
        X = integrate(x0, u_L, u_Q, p)
        S, I, Q = X[:, 0], X[:, 1], X[:, 2]

        # --- backward costates (lambda(T) = 0), Heun / trapezoidal -----------
        # costate ODE:  lambda' = rho*lambda - dH/dx, integrated backward.
        # The state (S, I, Q) and controls are frozen during this sweep, so the
        # derivative at each node is a closed function of the costate there. A
        # trapezoidal (Heun) step — Euler predictor + corrector at node k-1 —
        # matches the O(dt^2) forward pass, instead of the old O(dt) Euler that
        # left the adjoint one order less accurate than the state. Kept inline
        # (no per-step function call) so the interactive re-solve stays fast.
        lS[n] = lI[n] = lQ[n] = 0.0
        for k in range(n, 0, -1):
            # derivative at node k
            A = p.beta * (1.0 - u_L[k])
            h = max(0.0, I[k] - p.I_cap)
            d1S = p.rho * lS[k] - A * I[k] * (lI[k] - lS[k])
            d1I = p.rho * lI[k] - (
                p.w_I + p.V_D * p.mu + p.kappa * h - A * S[k] * lS[k]
                + lI[k] * (A * S[k] - (p.gamma + p.mu + u_Q[k]))
                + lQ[k] * u_Q[k]
            )
            d1Q = p.rho * lQ[k] - (
                p.w_Q + p.V_D * p.mu_Q - lQ[k] * (p.gamma_Q + p.mu_Q)
            )
            # Euler predictor one node back (backward step subtracts dt)
            pS = lS[k] - dt * d1S
            pI = lI[k] - dt * d1I
            pQ = lQ[k] - dt * d1Q
            # derivative at node k-1 evaluated on the predictor
            Ab = p.beta * (1.0 - u_L[k - 1])
            hb = max(0.0, I[k - 1] - p.I_cap)
            d2S = p.rho * pS - Ab * I[k - 1] * (pI - pS)
            d2I = p.rho * pI - (
                p.w_I + p.V_D * p.mu + p.kappa * hb - Ab * S[k - 1] * pS
                + pI * (Ab * S[k - 1] - (p.gamma + p.mu + u_Q[k - 1]))
                + pQ * u_Q[k - 1]
            )
            d2Q = p.rho * pQ - (
                p.w_Q + p.V_D * p.mu_Q - pQ * (p.gamma_Q + p.mu_Q)
            )
            # trapezoidal correction
            lS[k - 1] = lS[k] - 0.5 * dt * (d1S + d2S)
            lI[k - 1] = lI[k] - 0.5 * dt * (d1I + d2I)
            lQ[k - 1] = lQ[k] - 0.5 * dt * (d1Q + d2Q)

        # --- control update (closed form + damping) --------------------------
        uL_new = np.clip(p.beta * S * I * (lI - lS) / p.c_L, 0.0, p.u_L_max)
        uQ_new = np.clip(I * (lI - lQ) / p.c_Q, 0.0, p.u_Q_max)

        delta = max(np.max(np.abs(uL_new - u_L)), np.max(np.abs(uQ_new - u_Q)))
        u_L = (1.0 - damp) * u_L + damp * uL_new
        u_Q = (1.0 - damp) * u_Q + damp * uQ_new

        if delta < tol:
            converged = True
            break

    # Converged controls are now fixed, so the reported trajectory is a pure
    # forward IVP — integrate it once with RK4 for the highest-fidelity curve
    # that gets plotted and animated.
    X = integrate_rk4(x0, u_L, u_Q, p)
    S, I, Q, R, D = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    Reff = r_eff(S, u_L, u_Q, p)

    return Trajectory(t=t, S=S, I=I, Q=Q, R=R, D=D, u_L=u_L, u_Q=u_Q,
                      Reff=Reff, params=p, iters=it, converged=converged)


def cost_breakdown(traj: Trajectory) -> dict:
    """Decompose the realised objective J into its economic components."""
    p = traj.params
    dt = p.dt
    I, Q, uL, uQ = traj.I, traj.Q, traj.u_L, traj.u_Q
    over = np.maximum(0.0, I - p.I_cap)
    burden = float(np.sum(p.w_I * I + p.w_Q * Q) * dt)
    deaths = float(np.sum(p.V_D * (p.mu * I + p.mu_Q * Q)) * dt)
    lock = float(np.sum(0.5 * p.c_L * uL ** 2) * dt)
    quar = float(np.sum(0.5 * p.c_Q * uQ ** 2) * dt)
    cap = float(np.sum(0.5 * p.kappa * over ** 2) * dt)
    return {"burden": burden, "deaths": deaths, "lockdown": lock,
            "quarantine": quar, "capacity": cap,
            "total": burden + deaths + lock + quar + cap}


# ---------------------------------------------------------------------------
# Self-test / tuning probe
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = Params()
    print(f"R0 = {r0(p):.2f}   I_cap = {p.I_cap}")

    # no-control baseline
    n = p.n_steps
    base = integrate(
        np.array([0.999, 0.001, 0.0, 0.0, 0.0]),
        np.zeros(n + 1), np.zeros(n + 1), p)
    print(f"[no control]  peak I = {base[:,1].max():.3f}   "
          f"final D = {base[-1,4]:.4f}   final R = {base[-1,3]:.3f}")

    traj = solve(p)
    print(f"[optimal]     converged={traj.converged} in {traj.iters} iters")
    print(f"              peak I = {traj.I.max():.3f}   "
          f"peak Q = {traj.Q.max():.3f}   final D = {traj.D[-1]:.4f}")
    print(f"              u_L range [{traj.u_L.min():.2f}, {traj.u_L.max():.2f}]"
          f"   u_Q range [{traj.u_Q.min():.2f}, {traj.u_Q.max():.2f}]")
    print(f"              Reff range [{traj.Reff.min():.2f}, "
          f"{traj.Reff.max():.2f}]")
    # how long is I held under capacity?
    over = (traj.I > p.I_cap).sum() / len(traj.I)
    over_base = (base[:, 1] > p.I_cap).mean()
    print(f"              fraction of time I>I_cap: optimal={over:.2f} "
          f"baseline={over_base:.2f}")
    deaths_saved = base[-1, 4] - traj.D[-1]
    print(f"              deaths averted (fraction of pop): {deaths_saved:.4f}")
