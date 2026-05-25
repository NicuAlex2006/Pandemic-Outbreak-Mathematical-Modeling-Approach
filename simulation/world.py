import math
import numpy as np
import simulation.agent as _agent
from simulation.agent import (
    Agents, N, N_COMMUNITIES, COMMUNITY_SIZE, COMMUNITY_AREA, COMMUNITY_PADDING,
    BETA_SIM, GAMMA_SIM, MOVE_SPEED, DT,
    QUARANTINE_RECT,
)


class World:
    """
    Owns the agent arrays and advances the simulation one timestep at a time.
    Writes aggregate S/I/R history into shared_state (thread-safe via lock).
    """

    def __init__(self, shared_state: dict):
        self.agents = Agents()
        self.state = shared_state
        self.t_days: float = 0.0
        self.u_current: float = 0.0
        self.community_centers: np.ndarray = None  # (N_COMMUNITIES, 2) float32

    def setup(self):
        # Rectangular layout, communities laid out in a grid inside the community area.
        # COMMUNITY_AREA[2] is the number of rows.
        # Keeps all communities away from the quarantine zone (bottom-right).
        centers = []
        rows = COMMUNITY_AREA[2]
        for i in range(N_COMMUNITIES):
            row = i % rows
            column = i // rows
            centers.append([COMMUNITY_AREA[0] + column * (COMMUNITY_PADDING[0] + COMMUNITY_SIZE[0]),
                            COMMUNITY_AREA[1] + row * (COMMUNITY_PADDING[1] + COMMUNITY_SIZE[1])])
        self.community_centers = np.array(centers, dtype=np.float32)
        self.agents.initialize(self.community_centers)

    # --- main step --------------------------------------------------------

    def step(self):
        a = self.agents
        active = (a.statuses == Agents.I) | (a.statuses == Agents.Q)
        a.days_infected[active] += DT

        self._move()
        self._jump()
        self._infect()
        self._quarantine()
        self._die()
        self._recover()
        self._lose_immunity()
        self._aggregate()
        self.t_days += DT

    # --- sub-steps --------------------------------------------------------

    def _move(self):
        a = self.agents
        mobile = (a.statuses != Agents.Q) & (a.statuses != Agents.D)

        displacement = (np.random.randn(a.n, 2) * MOVE_SPEED).astype(np.float32)
        a.positions[mobile] += displacement[mobile]

        # keep agents within their community rect
        for c in range(N_COMMUNITIES):
            mask = mobile & (a.communities == c)
            if not mask.any():
                continue
            x0, y0 = self.community_centers[c]
            x1 = x0 + COMMUNITY_SIZE[0]
            y1 = y0 + COMMUNITY_SIZE[1]
            a.positions[mask, 0] = np.clip(a.positions[mask, 0], x0, x1)
            a.positions[mask, 1] = np.clip(a.positions[mask, 1], y0, y1)

    def _jump(self):
        a = self.agents
        mobile = (a.statuses != Agents.Q) & (a.statuses != Agents.D)
        roll = np.random.random(a.n)
        jumpers = mobile & (roll < _agent.JUMP_PROB)
        if not jumpers.any():
            return

        n_j = jumpers.sum()
        direction = np.random.choice([-1, 1], size=n_j)
        new_c = (a.communities[jumpers] + direction) % N_COMMUNITIES
        a.communities[jumpers] = new_c

        # Place inside the new community rect
        origins = self.community_centers[new_c]
        x0 = origins[:, 0]
        y0 = origins[:, 1]
        a.positions[jumpers, 0] = np.random.uniform(x0 + 1, x0 + COMMUNITY_SIZE[0] - 1, n_j)
        a.positions[jumpers, 1] = np.random.uniform(y0 + 1, y0 + COMMUNITY_SIZE[1] - 1, n_j)

    def _infect(self):
        a = self.agents
        newly_infected = np.zeros(a.n, dtype=bool)

        for c in range(N_COMMUNITIES):
            i_mask = (a.statuses == Agents.I) & (a.communities == c)
            s_mask = (a.statuses == Agents.S) & (a.communities == c)
            if not i_mask.any() or not s_mask.any():
                continue

            i_pos = a.positions[i_mask]  # (n_i, 2)
            s_pos = a.positions[s_mask]  # (n_s, 2)

            # (n_i, n_s) pairwise distances
            diff = s_pos[np.newaxis, :, :] - i_pos[:, np.newaxis, :]
            dists = np.linalg.norm(diff, axis=2)

            n_contacts = (dists < _agent.INFECTION_RADIUS).sum(axis=0)  # (n_s,)
            effective_beta = BETA_SIM * (1.0 - self.u_current)
            p_survive = (1.0 - effective_beta * DT) ** n_contacts

            s_idx = np.where(s_mask)[0]
            roll = np.random.random(len(s_idx))
            newly_infected[s_idx[roll > p_survive]] = True

        a.statuses[newly_infected] = Agents.I

    def _quarantine(self):
        if self.u_current <= 0:
            return
        a = self.agents
        eligible = (a.statuses == Agents.I) & (a.days_infected > _agent.QUARANTINE_DELAY)
        roll = np.random.random(a.n)
        to_q = eligible & (roll < self.u_current * DT)
        if not to_q.any():
            return

        a.statuses[to_q] = Agents.Q
        qx, qy, qw, qh = QUARANTINE_RECT
        n_q = int(to_q.sum())
        a.positions[to_q] = np.column_stack([
            np.random.uniform(qx + 10, qx + qw - 10, n_q),
            np.random.uniform(qy + 10, qy + qh - 10, n_q),
        ]).astype(np.float32)

    def _die(self):
        a = self.agents
        can_die = (a.statuses == Agents.I) | (a.statuses == Agents.Q)
        mu = self.state.get("mu_fit", 0.003)
        roll = np.random.random(a.n)
        died = can_die & (roll < mu * DT)
        a.statuses[died] = Agents.D
        a.days_infected[died] = 0.0

    def _recover(self):
        a = self.agents
        can_recover = (a.statuses == Agents.I) | (a.statuses == Agents.Q)
        roll = np.random.random(a.n)
        recovered = can_recover & (roll < GAMMA_SIM * DT)
        a.statuses[recovered] = Agents.R
        a.days_infected[recovered] = 0.0

    def _lose_immunity(self):
        a = self.agents
        omega = self.state.get("omega", 0.0)
        if omega <= 0:
            return
        recovered = a.statuses == Agents.R
        roll = np.random.random(a.n)
        lost = recovered & (roll < omega * DT)
        a.statuses[lost] = Agents.S

    def _aggregate(self):
        S, I, R, D = self.agents.sird_fractions()
        with self.state['lock']:
            self.state['S_history'].append(S)
            self.state['I_history'].append(I)
            self.state['R_history'].append(R)
            self.state.setdefault('D_history', []).append(D)
            self.state['t_history'].append(self.t_days)
            self.state['t_days'] = self.t_days
            self.state['u_current'] = self.u_current

    # --- helpers ----------------------------------------------------------

    def set_u(self, u: float):
        self.u_current = float(np.clip(u, 0.0, 1.0))

    def get_sir(self) -> tuple[float, float, float]:
        return self.agents.sir_fractions()
