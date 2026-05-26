"""
Entry point: fit SIRD-S parameters from French COVID data at startup,
then run the pygame simulation with a live dashboard.

Startup (ONCE): fetch_france_data → fit_sird → solve_hjb → solve_binomial
Per-alpha (worker): solve_hjb only (β,γ,μ frozen)
Per-frame (main): u* lookup + ABM step + dashboard blit
"""

import threading
import numpy as np

from simulation.world import World
from simulation.renderer import Renderer
from dashboard import Dashboard
from models.adapter import (solve_hjb, solve_binomial_stopping,
                            fit_sird, fetch_france_data)

T_SIM = 365.0
ALPHA_I_DEFAULT = 1.0
ALPHA_D_DEFAULT = 1000.0
I_CAP_DEFAULT = 0.03
W_H_DEFAULT = 500.0
S0_DEFAULT = 0.80
I0_DEFAULT = 0.20


def main():
    # =================================================================
    # STARTUP — fit from real French COVID data, results frozen
    # =================================================================
    print("=== Startup: loading French COVID data ===")
    try:
        t_data, I_data, D_data, N_eff = fetch_france_data()
        print(f"  {len(t_data)} days, N_eff={N_eff:.0f}")
        print(f"  I peak={I_data.max():.4f}, D final={D_data[-1]:.6f}")
    except Exception as e:
        print(f"  OWID fetch failed ({e}), using synthetic data")
        t_data = np.linspace(0, 10, 50)
        I_data = I0_DEFAULT * np.exp(0.15 * t_data)
        I_data = np.clip(I_data, 0, 0.5)
        D_data = 0.02 * np.cumsum(I_data) * (t_data[1] - t_data[0])
        D_data = np.clip(D_data, 0, 0.1)

    print("=== Startup: fitting β, γ, μ (runs ONCE) ===")
    beta, gamma, mu = fit_sird(I_data, D_data, S0_DEFAULT, I0_DEFAULT,
                                T=t_data[-1])
    cfr = mu / (mu + gamma) if (mu + gamma) > 0 else 0.0
    print(f"  β={beta:.4f}  γ={gamma:.4f}  μ={mu:.6f}")
    print(f"  CFR = {cfr*100:.2f}%")

    print("=== Startup: initial HJB solve ===")
    V, u_opt, S_grid, I_grid, g_array, stopped_frac = solve_hjb(
        beta, gamma, ALPHA_I_DEFAULT, alpha_d=ALPHA_D_DEFAULT,
        T=T_SIM, mu=mu, i_cap=I_CAP_DEFAULT, w_h=W_H_DEFAULT,
    )
    print(f"  V range: [{V.min():.3f}, {V.max():.3f}]  u_opt: {u_opt.shape}")
    print(f"  Stopping: {'Yes' if stopped_frac > 0.01 else 'No'} ({stopped_frac:.1%})")

    print("=== Startup: binomial stopping tree ===")
    _, stop_tree, betas_tree = solve_binomial_stopping(beta, gamma, I0_DEFAULT)
    print(f"  Nodes: {len(stop_tree)}")
    print("=== Startup complete ===\n")

    from simulation.agent import N as N_AGENTS
    n_initial_infected = 5
    I0_sim = n_initial_infected / N_AGENTS
    S0_sim = 1.0 - I0_sim

    shared_state = {
        "S_history": [], "I_history": [], "R_history": [],
        "D_history": [], "t_history": [],
        "t_days": 0.0, "u_current": 0.0,
        "beta_fit": beta, "gamma_fit": gamma, "mu_fit": mu,
        "u_opt": u_opt, "S_grid": S_grid, "I_grid": I_grid,
        "alpha_i": ALPHA_I_DEFAULT, "alpha_d": ALPHA_D_DEFAULT,
        "i_cap": I_CAP_DEFAULT, "w_h": W_H_DEFAULT,
        "omega": 0.0, "I0": I0_sim, "S0": S0_sim,
        "lock": threading.Lock(), "hjb_running": False,
    }

    dashboard = Dashboard(
        beta=beta, gamma=gamma, mu=mu,
        V=V, u_opt=u_opt, g_array=g_array,
        S_grid=S_grid, I_grid=I_grid,
        alpha_i=ALPHA_I_DEFAULT, alpha_d=ALPHA_D_DEFAULT,
        i_cap=I_CAP_DEFAULT, w_h=W_H_DEFAULT,
        T=T_SIM, stopped_frac=stopped_frac,
    )

    world = World(shared_state=shared_state)
    world.setup()
    renderer = Renderer(world=world, shared_state=shared_state,
                        dashboard=dashboard)
    renderer.setup()
    renderer.run()


if __name__ == "__main__":
    main()
