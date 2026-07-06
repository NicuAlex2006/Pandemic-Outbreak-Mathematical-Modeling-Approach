import numpy as np
from models.adapter import get_quarantine_u_at


class ControlBridge:
    """
    Translates the (I, Q) Bellman policy surface into a per-step quarantine
    intensity. Applies an exponential moving average to avoid abrupt jumps.
    """

    def __init__(self, ema_alpha: float = 0.25):
        self.ema_alpha = ema_alpha      # smoothing factor ∈ (0, 1]
        self.u_smooth: float = 0.0     # current smoothed control value

    def get_quarantine_prob(
        self,
        I: float,
        Q: float,
        policy: np.ndarray,
        I_axis: np.ndarray,
        Q_axis: np.ndarray,
        manual_override: float = None,
    ) -> float:
        """
        Return quarantine intensity u*(I, Q) for the current sim state.
        If manual_override is in [0,1] it bypasses the policy.
        Otherwise: u_raw = get_quarantine_u_at(...), then EMA smoothing.
        """
        if manual_override is not None:
            return float(np.clip(manual_override, 0.0, 1.0))
        if policy is None:
            return 0.0
        u_raw = get_quarantine_u_at(I, Q, policy, I_axis, Q_axis)
        self.u_smooth = self.ema_alpha * u_raw + (1.0 - self.ema_alpha) * self.u_smooth
        return float(np.clip(self.u_smooth, 0.0, 1.0))

    def reset(self):
        """Reset smoothed state (call when the policy is recomputed)."""
        self.u_smooth = 0.0
