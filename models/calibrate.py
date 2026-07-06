"""
Calibrate the epidemic rates (beta, gamma, mu) from France's first COVID wave.

Design choice (consistent, normalisation-free):
  * gamma  = 1 / infectious_period          (fixed from clinical period)
  * beta   = r + gamma, where r is the early *exponential growth rate* of cases
             (during the initial phase S ~ 1, so dI/dt ~ (beta - gamma) I).
             This recovers the natural, pre-intervention R0 rather than the
             lockdown-suppressed curve.
  * mu     = gamma * IFR / (1 - IFR)         from a target infection-fatality
             ratio (the confirmed-case CFR is unreliable due to under-testing).

Returns clean rates regardless of data quirks; the UI sliders let the user
override any of them.
"""

from __future__ import annotations

import numpy as np

from models.sir import _load_owid_france


def _early_growth_rate(new_cases: np.ndarray, window: int = 21) -> float:
    """Log-linear slope of smoothed new cases over the early rising phase."""
    c = np.asarray(new_cases, dtype=float)
    c = c[np.isfinite(c)]
    # onset = first day above 1% of the eventual peak
    peak = c.max() if c.size else 1.0
    onset = np.argmax(c >= 0.01 * peak) if peak > 0 else 0
    seg = c[onset:onset + window]
    seg = seg[seg > 0]
    if seg.size < 5:
        return 0.2  # fallback growth rate
    t = np.arange(seg.size, dtype=float)
    slope = np.polyfit(t, np.log(seg), 1)[0]
    return float(slope)


def calibrate_from_france(infectious_period: float = 8.0,
                          ifr: float = 0.01) -> dict:
    """Return {'beta','gamma','mu','R0','r'} fitted to the France wave."""
    gamma = 1.0 / infectious_period
    try:
        df = _load_owid_france()
        new_cases = df["new_cases"].fillna(0).rolling(7, min_periods=1).mean()
        r = _early_growth_rate(new_cases.values)
    except Exception:
        r = 0.2
    beta = max(r + gamma, 1.05 * gamma)       # keep R0 > 1
    mu = gamma * ifr / (1.0 - ifr)
    return {"beta": beta, "gamma": gamma, "mu": mu,
            "R0": beta / (gamma + mu), "r": r}


if __name__ == "__main__":
    res = calibrate_from_france()
    print("Calibration from France first wave:")
    for k, v in res.items():
        print(f"  {k:6s} = {v:.4f}")
