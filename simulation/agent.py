"""
Agent population — a *visual sampler* of the SIRDQ ODE.

The agents carry no epidemic dynamics of their own: the World relabels them
each frame so the on-screen S/I/Q/R/D counts track the optimal ODE trajectory
exactly. This module only owns the arrays, the spatial layout, and counting.

Status codes: 0=S, 1=I, 2=Q, 3=R, 4=D
"""

import numpy as np

# ── population & layout ──────────────────────────────────────────────────────
N = 6000
N_COMMUNITIES = 20
GRID_COLS = 5
GRID_ROWS = 4
COMMUNITY_SIZE = (105, 105)      # px
COMMUNITY_PADDING = (12, 12)     # px
COMMUNITY_ORIGIN = (24, 92)      # px (top-left of the community grid)

QUARANTINE_RECT = (650, 92, 230, 300)   # (x, y, w, h) px

MOVE_SPEED = 1.4                 # px/frame jitter at zero lockdown


class Agents:
    S, I, Q, R, D = 0, 1, 2, 3, 4

    def __init__(self, n: int = N, n_communities: int = N_COMMUNITIES):
        self.n = n
        self.n_communities = n_communities
        self.positions = None        # (N, 2) float32
        self.communities = None      # (N,)   int32
        self.statuses = None         # (N,)   int8
        self.days_infected = None    # (N,)   float32  (for ageing/ordering)

    # --- setup ----------------------------------------------------------------

    def community_centers(self) -> np.ndarray:
        centers = []
        ox, oy = COMMUNITY_ORIGIN
        for c in range(self.n_communities):
            row, col = divmod(c, GRID_COLS)
            centers.append([ox + col * (COMMUNITY_SIZE[0] + COMMUNITY_PADDING[0]),
                            oy + row * (COMMUNITY_SIZE[1] + COMMUNITY_PADDING[1])])
        return np.array(centers, dtype=np.float32)

    def initialize(self, centers: np.ndarray, n_initial_infected: int = 8):
        n = self.n
        self.communities = np.random.randint(0, self.n_communities, n).astype(np.int32)
        off = np.column_stack([
            COMMUNITY_SIZE[0] * np.random.random(n),
            COMMUNITY_SIZE[1] * np.random.random(n),
        ]).astype(np.float32)
        self.positions = (centers[self.communities] + off).astype(np.float32)
        self.statuses = np.zeros(n, dtype=np.int8)
        self.days_infected = np.zeros(n, dtype=np.float32)
        if n_initial_infected > 0:
            seed = np.random.choice(n, size=min(n_initial_infected, n), replace=False)
            self.statuses[seed] = Agents.I

    # --- counts ---------------------------------------------------------------

    def counts(self) -> tuple[int, int, int, int, int]:
        s = self.statuses
        return (int((s == Agents.S).sum()), int((s == Agents.I).sum()),
                int((s == Agents.Q).sum()), int((s == Agents.R).sum()),
                int((s == Agents.D).sum()))

    def fractions(self) -> tuple[float, float, float, float, float]:
        return tuple(c / self.n for c in self.counts())
