"""
Thin adapter exposing the contract interface the simulation expects.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from models.sir import (
    euler_forward as _euler_forward,
    fit_sir as _fit_sir,
    fit_sird as _fit_sird,
    sirds_euler_forward as _sirds_euler_forward,
    fetch_france_data as _fetch_france_data,
)
from models.hjb import solve_hjb as _solve_hjb
from models.stopping import (
    build_beta_tree as _build_beta_tree,
    compute_cost as _compute_cost,
    snell_envelope as _snell_envelope,
)


def sir_euler(beta, gamma, S0, I0, T, n):
    t = np.linspace(0, T, n)
    S, I, R = _euler_forward(S0, I0, 0.0, beta, gamma, t)
    return S, I, R


def sirds_euler(beta, gamma, mu, S0, I0, T, n, omega=0.0, u=0.0):
    t = np.linspace(0, T, n)
    S, I, R, D = _sirds_euler_forward(S0, I0, 0.0, 0.0, beta, gamma, mu, t,
                                       omega=omega, u=u)
    return S, I, R, D, t


def fit_sir(I_data, S0, I0, T):
    t = np.arange(len(I_data), dtype=float)
    if T > 0 and len(I_data) > 1:
        t = t * (T / (len(I_data) - 1))
    beta, gamma = _fit_sir(t, S0, I0, 0.0, I_data)
    return beta, gamma


def fit_sird(I_data, D_data, S0, I0, T):
    t = np.arange(len(I_data), dtype=float)
    if T > 0 and len(I_data) > 1:
        t = t * (T / (len(I_data) - 1))
    beta, gamma, mu = _fit_sird(t, S0, I0, 0.0, 0.0, I_data, D_data)
    return beta, gamma, mu


def fetch_france_data():
    return _fetch_france_data()


def solve_hjb(beta, gamma, alpha_i, alpha_d=1000.0, T=10.0,
              nS=40, nI=40, n_time=100, mu=0.0,
              i_cap=0.03, w_h=500.0):
    s, i, V, u_opt_2d, g_array, _snaps, stopped_frac = _solve_hjb(
        beta=beta, gamma=gamma, alpha_i=alpha_i, alpha_d=alpha_d,
        T=T, nS=nS, nI=nI, mu=mu, i_cap=i_cap, w_h=w_h,
    )
    u_opt = np.broadcast_to(u_opt_2d, (n_time, nS, nI)).copy()
    return V, u_opt, s, i, g_array, stopped_frac


def get_u_at(S_val, I_val, t_idx, u_opt, S_grid, I_grid):
    if u_opt is None:
        return 0.0
    t_idx = min(t_idx, u_opt.shape[0] - 1)
    interp = RegularGridInterpolator(
        (S_grid, I_grid), u_opt[t_idx],
        method="linear", bounds_error=False, fill_value=0.0,
    )
    return float(np.asarray(interp([[S_val, I_val]])).flat[0])


def solve_binomial_stopping(beta0, gamma, I0, p=0.6, u_factor=1.3,
                            d_factor=0.8, g_func=None, n=8):
    beta_tree = _build_beta_tree(beta0, u_factor, d_factor, n)
    g_tree = _compute_cost(beta_tree, n)
    U_arr, stop_mask = _snell_envelope(g_tree, n, p)
    U_dict, stop_dict, betas_dict = {}, {}, {}
    for k in range(n + 1):
        for j in range(k + 1):
            U_dict[(k, j)] = float(U_arr[j, k])
            stop_dict[(k, j)] = bool(stop_mask[j, k])
            betas_dict[(k, j)] = float(beta_tree[j, k])
    return U_dict, stop_dict, betas_dict
