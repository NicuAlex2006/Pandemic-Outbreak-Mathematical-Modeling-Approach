import math
import numpy as np

# Simulation parameters
N = 8000
N_COMMUNITIES = 20
COMMUNITY_SIZE = (100,100)    # pixels
COMMUNITY_AREA = (150, 100, 3)  # (x, y, r) in pixels
COMMUNITY_PADDING = (5, 5) # pixels

INFECTION_RADIUS = 4    # pixels
BETA_SIM = 0.2           # transmission prob per timestep per nearby infected
GAMMA_SIM = 0.09         # recovery prob per timestep
MOVE_SPEED = 1.5         # pixels per timestep
JUMP_PROB = 0.007        # prob of jumping to adjacent community per step
QUARANTINE_DELAY = 3.0   # simulated days before quarantine eligible
DT = 0.1                 # timestep in simulated days

QUARANTINE_RECT = (650, 500, 220, 160)  # (x, y, w, h) in pixels


class Agents:
    """
    Fully vectorized agent state stored as numpy arrays.
    Status codes: 0=Susceptible, 1=Infected, 2=Recovered, 3=Quarantined, 4=Dead
    """

    S, I, R, Q, D = 0, 1, 2, 3, 4

    def __init__(self, n: int = N, n_communities: int = N_COMMUNITIES):
        self.n = n
        self.n_communities = n_communities
        self.positions: np.ndarray = None       # (N, 2) float32
        self.communities: np.ndarray = None     # (N,)   int32
        self.statuses: np.ndarray = None        # (N,)   int32
        self.days_infected: np.ndarray = None   # (N,)   float32

    def initialize(self, community_centers: np.ndarray, n_initial_infected: int = 5):
        n = self.n
        # Random community assignment
        self.communities = np.random.randint(0, self.n_communities, size=n).astype(np.int32)

        # Uniform random placement within each community rect
        x = COMMUNITY_SIZE[0] * np.random.random(n).astype(np.float32)
        y = COMMUNITY_SIZE[1] * np.random.random(n).astype(np.float32)
        offsets = np.column_stack([x, y]).astype(np.float32)
        self.positions = (community_centers[self.communities] + offsets).astype(np.float32)

        self.statuses = np.zeros(n, dtype=np.int32)
        self.days_infected = np.zeros(n, dtype=np.float32)

        infected_idx = np.random.choice(n, size=n_initial_infected, replace=False)
        self.statuses[infected_idx] = Agents.I

    # --- convenience masks ------------------------------------------------

    @property
    def susceptible(self) -> np.ndarray:
        return self.statuses == Agents.S

    @property
    def infected(self) -> np.ndarray:
        return self.statuses == Agents.I

    @property
    def recovered(self) -> np.ndarray:
        return self.statuses == Agents.R

    @property
    def quarantined(self) -> np.ndarray:
        return self.statuses == Agents.Q

    # --- aggregate fractions ----------------------------------------------

    @property
    def dead(self) -> np.ndarray:
        return self.statuses == Agents.D

    def sir_fractions(self) -> tuple[float, float, float]:
        """(S, I, R) fractions of N. Q agents count toward I."""
        n = self.n
        S = float(self.susceptible.sum()) / n
        I = float((self.infected.sum() + self.quarantined.sum())) / n
        R = float(self.recovered.sum()) / n
        return S, I, R

    def sird_fractions(self):
        """(S, I, R, D) fractions."""
        n = self.n
        S = float(self.susceptible.sum()) / n
        I = float((self.infected.sum() + self.quarantined.sum())) / n
        R = float(self.recovered.sum()) / n
        D = float(self.dead.sum()) / n
        return S, I, R, D
