"""
World — advances a playhead along the optimal ODE trajectory and relabels the
agent population so the on-screen S/I/Q/R/D counts match it exactly.

The epidemic lives entirely in the ODE; the agents are a faithful particle
rendering of it (mass is conserved, so counts always reconcile).
"""

import numpy as np

from simulation.agent import (
    Agents, N_COMMUNITIES, COMMUNITY_SIZE, QUARANTINE_RECT, MOVE_SPEED,
)


def _round_counts(fracs, N: int) -> np.ndarray:
    """Largest-remainder rounding so integer counts sum exactly to N."""
    raw = np.asarray(fracs, dtype=float) * N
    base = np.floor(raw).astype(int)
    deficit = N - int(base.sum())
    if deficit != 0:
        rem = raw - base
        order = np.argsort(-rem) if deficit > 0 else np.argsort(rem)
        base[order[:abs(deficit)]] += np.sign(deficit)
    return base


class World:
    def __init__(self, shared_state: dict | None = None):
        self.agents = Agents()
        self.state = shared_state or {}
        self.centers = None
        self.trajectory = None
        self.t_days = 0.0
        self.u_L = 0.0
        self.u_Q = 0.0

    # --- setup ----------------------------------------------------------------

    def setup(self, n_initial_infected: int = 8):
        self.centers = self.agents.community_centers()
        self.agents.initialize(self.centers, n_initial_infected)

    def reset(self, n_initial_infected: int = 8):
        self.agents.initialize(self.centers, n_initial_infected)
        self.t_days = 0.0
        self.u_L = self.u_Q = 0.0

    def set_trajectory(self, traj, reset_playhead: bool = False):
        self.trajectory = traj
        if reset_playhead:
            self.t_days = 0.0

    def finished(self) -> bool:
        return self.trajectory is not None and self.t_days >= self.trajectory.t[-1]

    # --- main step ------------------------------------------------------------

    def step(self, dt_days: float):
        if self.trajectory is None:
            return None
        self.t_days = min(self.t_days + dt_days, self.trajectory.t[-1])
        smp = self.trajectory.at(self.t_days)
        self.u_L, self.u_Q = smp["u_L"], smp["u_Q"]

        targets = _round_counts(
            [smp["S"], smp["I"], smp["Q"], smp["R"], smp["D"]], self.agents.n)
        self._reconcile(targets)
        self._age(dt_days)
        self._animate(smp["u_L"])
        return smp

    # --- reconciliation (relabel agents to hit target counts) ----------------

    def _reconcile(self, targets):
        a = self.agents
        s = a.statuses
        nS, nI, nQ, nR, nD = (int(x) for x in targets)
        A = Agents

        # 1. deaths (absorbing) — drawn from the infectious pool, oldest first
        need = nD - int((s == A.D).sum())
        if need > 0:
            self._resolve_oldest((s == A.I) | (s == A.Q), need, A.D)

        # 2. recoveries (absorbing) — likewise
        need = nR - int((s == A.R).sum())
        if need > 0:
            self._resolve_oldest((s == A.I) | (s == A.Q), need, A.R)

        # 3. new infections S -> I, biased toward communities already infected
        need = int((s == A.S).sum()) - nS
        if need > 0:
            self._infect(need)

        # 4. split the infectious pool to match the quarantine target
        cQ = int((s == A.Q).sum())
        if cQ < nQ:
            self._isolate(nQ - cQ)
        elif cQ > nQ:
            self._release(cQ - nQ)

    def _resolve_oldest(self, mask, need, new_status):
        a = self.agents
        pool = np.where(mask)[0]
        if pool.size == 0:
            return
        order = pool[np.argsort(-a.days_infected[pool])]
        pick = order[:need]
        a.statuses[pick] = new_status
        a.days_infected[pick] = 0.0

    def _infect(self, need):
        a = self.agents
        s_idx = np.where(a.statuses == Agents.S)[0]
        if s_idx.size == 0:
            return
        need = min(need, s_idx.size)
        inf_by_comm = np.bincount(
            a.communities[a.statuses == Agents.I],
            minlength=N_COMMUNITIES).astype(float)
        w = 0.15 + inf_by_comm[a.communities[s_idx]]
        w /= w.sum()
        pick = np.random.choice(s_idx, size=need, replace=False, p=w)
        a.statuses[pick] = Agents.I
        a.days_infected[pick] = 0.0

    def _isolate(self, need):
        a = self.agents
        i_idx = np.where(a.statuses == Agents.I)[0]
        if i_idx.size == 0:
            return
        need = min(need, i_idx.size)
        # detect/isolate the longest-infected first
        order = i_idx[np.argsort(-a.days_infected[i_idx])]
        pick = order[:need]
        a.statuses[pick] = Agents.Q
        qx, qy, qw, qh = QUARANTINE_RECT
        a.positions[pick] = np.column_stack([
            np.random.uniform(qx + 8, qx + qw - 8, need),
            np.random.uniform(qy + 8, qy + qh - 8, need),
        ]).astype(np.float32)

    def _release(self, need):
        a = self.agents
        q_idx = np.where(a.statuses == Agents.Q)[0]
        if q_idx.size == 0:
            return
        need = min(need, q_idx.size)
        pick = q_idx[:need]
        a.statuses[pick] = Agents.I
        ctr = self.centers[a.communities[pick]]
        a.positions[pick] = (ctr + np.column_stack([
            np.random.uniform(8, COMMUNITY_SIZE[0] - 8, need),
            np.random.uniform(8, COMMUNITY_SIZE[1] - 8, need),
        ])).astype(np.float32)

    # --- ageing & animation ---------------------------------------------------

    def _age(self, dt_days):
        a = self.agents
        active = (a.statuses == Agents.I) | (a.statuses == Agents.Q)
        a.days_infected[active] += dt_days

    def _animate(self, u_L):
        a = self.agents
        s = a.statuses
        amp = MOVE_SPEED * (1.0 - 0.92 * float(u_L))   # lockdown freezes movement
        mob = (s == Agents.S) | (s == Agents.I) | (s == Agents.R)
        if not mob.any():
            return
        disp = (np.random.randn(a.n, 2) * amp).astype(np.float32)
        a.positions[mob] += disp[mob]
        ctr = self.centers[a.communities]
        a.positions[mob, 0] = np.clip(a.positions[mob, 0],
                                      ctr[mob, 0], ctr[mob, 0] + COMMUNITY_SIZE[0])
        a.positions[mob, 1] = np.clip(a.positions[mob, 1],
                                      ctr[mob, 1], ctr[mob, 1] + COMMUNITY_SIZE[1])
