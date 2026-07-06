"""
Thin adapter: the only surface the simulation/UI layer touches.

  calibrate()      -> (Params, info)   rates from France data + tuned economics
  solve_optimal()  -> Trajectory       PMP optimal (u_L, u_Q) trajectory
"""

from __future__ import annotations

from dataclasses import replace

from models.sirdq import Params, r0
from models.pmp import solve as _solve, Trajectory
from models.calibrate import calibrate_from_france


def calibrate(min_R0: float = 3.2, ifr: float = 0.01,
              infectious_period: float = 8.0) -> tuple[Params, dict]:
    """Build Params from the France wave.

    gamma (clinical period) and mu (target IFR) come straight from the data
    assumptions. beta is set from an *unmitigated* R0: the France first wave was
    suppressed by the March-2020 lockdown, so its observed growth understates
    the natural R0. We surface both the fitted and scenario R0 in `info`.
    """
    cal = calibrate_from_france(infectious_period=infectious_period, ifr=ifr)
    R0_scenario = max(cal["R0"], min_R0)
    beta = R0_scenario * (cal["gamma"] + cal["mu"])
    params = Params(beta=beta, gamma=cal["gamma"], mu=cal["mu"])
    info = {
        "R0_fitted": cal["R0"],
        "R0_scenario": r0(params),
        "growth_rate": cal["r"],
        "suppressed": cal["R0"] < min_R0,
    }
    return params, info


def solve_optimal(params: Params, **kw) -> Trajectory:
    """Optimal two-control trajectory for the given parameters."""
    return _solve(params, **kw)


def with_overrides(params: Params, **overrides) -> Params:
    """Return a copy of params with fields replaced (used by UI sliders)."""
    return replace(params, **overrides)
