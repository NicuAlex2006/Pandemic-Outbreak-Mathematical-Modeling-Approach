"""
Optimal Epidemic Control — entry point.

  1. Calibrate epidemic rates from France's first COVID wave.
  2. Solve the two-control (lockdown + isolation) optimal-control problem (PMP).
  3. Launch the pygame view: agents render the optimal ODE trajectory live,
     and the parameter sliders re-solve the control on the fly.

The ODE is the single ground truth — the particle cloud, the trajectory plot,
and the policy are all the same solution.
"""

from models.adapter import calibrate, solve_optimal
from models.sirdq import r0
from simulation.world import World
from simulation.renderer import Renderer
from dashboard import Dashboard


def main():
    print("=== Calibrating from France first wave ===")
    params, info = calibrate()
    print(f"  fitted R0   = {info['R0_fitted']:.2f}  (growth r={info['growth_rate']:.3f}/day)")
    if info["suppressed"]:
        print(f"  note: first wave was lockdown-suppressed; using unmitigated "
              f"R0 = {info['R0_scenario']:.2f} for the baseline scenario")
    print(f"  beta={params.beta:.3f}  gamma={params.gamma:.3f}  "
          f"mu={params.mu:.4f}  R0={r0(params):.2f}")

    print("=== Solving optimal control (PMP forward-backward sweep) ===")
    traj = solve_optimal(params)
    print(f"  converged={traj.converged} in {traj.iters} iters")
    print(f"  peak I={traj.I.max():.3f} (cap {params.I_cap})  "
          f"final D={traj.D[-1]*100:.2f}%")
    print(f"  u_L max={traj.u_L.max():.2f}  u_Q max={traj.u_Q.max():.2f}  "
          f"R_eff min={traj.Reff.min():.2f}")
    print("=== Launching ===\n")

    world = World()
    world.setup(n_initial_infected=8)
    world.set_trajectory(traj, reset_playhead=True)

    dashboard = Dashboard(params, traj, info)

    renderer = Renderer(world, dashboard, params)
    renderer.setup()
    renderer.run()


if __name__ == "__main__":
    main()
