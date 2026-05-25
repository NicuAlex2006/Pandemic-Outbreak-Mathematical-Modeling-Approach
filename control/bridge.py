import numpy as np
from models.adapter import get_u_at


class ControlBridge:
    """
    Translates the HJB value function into a per-step quarantine probability.
    Applies exponential moving average to avoid abrupt policy jumps.
    """

    def __init__(self, ema_alpha: float = 0.1):
        self.ema_alpha = ema_alpha      # smoothing factor ∈ (0, 1]
        self.u_smooth: float = 0.0     # current smoothed control value

    def get_quarantine_prob(
        self,
        S: float,
        I: float,
        t_idx: int,
        u_opt: np.ndarray,
        S_grid: np.ndarray,
        I_grid: np.ndarray,
        manual_override: float = None,
    ) -> float:
        """
        Return quarantine probability for the current sim state.
        If manual_override is in [0,1] it bypasses the HJB policy.
        Otherwise: u_raw = get_u_at(...), then EMA smoothing.
        """
        if manual_override is not None:
            return float(np.clip(manual_override, 0.0, 1.0))
        if u_opt is None:
            return 0.0
        u_raw = get_u_at(S, I, t_idx, u_opt, S_grid, I_grid)
        self.u_smooth = self.ema_alpha * u_raw + (1.0 - self.ema_alpha) * self.u_smooth
        return float(np.clip(self.u_smooth, 0.0, 1.0))

    def reset(self):
        """Reset smoothed state (call when HJB solution is recomputed)."""
        self.u_smooth = 0.0
