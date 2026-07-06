"""
Dashboard — three Agg-rendered panels blitted into pygame.

 Panel 1  optimal trajectory S/I/Q/R/D vs t, with the hospital-capacity line
          and a live playhead.
 Panel 2  the two optimal controls u_L(t), u_Q(t) and R_eff(t) (target = 1).
 Panel 3  text stats + cost breakdown (the realised objective J).

A worker thread re-solves the PMP optimal control whenever the parameters
change, so the UI never blocks.
"""

import queue
import threading
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pygame

from models.adapter import solve_optimal
from models.pmp import cost_breakdown
from models.sirdq import r0

REFRESH_INTERVAL = 0.4


class Dashboard:
    def __init__(self, params, trajectory, info=None):
        self.params = params
        self.trajectory = trajectory
        self.info = info or {}

        self._worker_status = "idle"
        self._last_solve_ms = 0.0
        self.fps = 0.0

        self.req_q = queue.Queue(maxsize=1)
        self.res_q = queue.Queue()
        self.worker = threading.Thread(target=self._solver_loop, daemon=True)
        self.worker.start()

        self.fig = plt.figure(figsize=(5.6, 7.7), dpi=100)
        gs = self.fig.add_gridspec(
            3, 1, height_ratios=[1.15, 0.95, 0.85],
            hspace=0.42, left=0.12, right=0.96, top=0.96, bottom=0.05)
        self.ax_traj = self.fig.add_subplot(gs[0])
        self.ax_ctrl = self.fig.add_subplot(gs[1])
        self.ax_stats = self.fig.add_subplot(gs[2])
        self.ax_stats.axis("off")
        self.ax_reff = self.ax_ctrl.twinx()

        self.canvas = FigureCanvasAgg(self.fig)
        self.surface = None
        self._last_render = 0.0

    # --- worker --------------------------------------------------------------

    def _solver_loop(self):
        while True:
            req = self.req_q.get()
            if req is None:
                return
            self._worker_status = "solving…"
            t0 = time.monotonic()
            try:
                traj = solve_optimal(req)
                self._last_solve_ms = (time.monotonic() - t0) * 1e3
                self.res_q.put(("ok", req, traj))
            except Exception as e:               # noqa: BLE001
                self.res_q.put(("err", req, e))
            self._worker_status = "idle"

    def request_solve(self, params):
        try:
            self.req_q.get_nowait()
        except queue.Empty:
            pass
        try:
            self.req_q.put_nowait(params)
        except queue.Full:
            pass

    def poll_worker(self):
        """Return a fresh Trajectory if the worker produced one, else None."""
        try:
            tag, params, payload = self.res_q.get_nowait()
        except queue.Empty:
            return None
        if tag == "err":
            self._worker_status = f"error: {payload}"
            return None
        self.params = params
        self.trajectory = payload
        return payload

    # --- render --------------------------------------------------------------

    def render_to_surface(self, t_days):
        now = time.monotonic()
        if self.surface is not None and now - self._last_render < REFRESH_INTERVAL:
            return self.surface
        self._last_render = now

        self._draw_trajectory(t_days)
        self._draw_controls(t_days)
        self._draw_stats(t_days)

        self.canvas.draw()
        buf = self.canvas.buffer_rgba()
        w, h = self.canvas.get_width_height()
        rgb = np.frombuffer(buf, np.uint8).reshape(h, w, 4)[:, :, :3].copy()
        self.surface = pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")
        return self.surface

    def _draw_trajectory(self, t_days):
        ax = self.ax_traj
        ax.clear()
        tr, p = self.trajectory, self.params
        ax.plot(tr.t, tr.S, color="#6495ED", lw=1.3, label="S")
        ax.plot(tr.t, tr.I, color="#DC3232", lw=1.6, label="I")
        ax.plot(tr.t, tr.Q, color="#FFA500", lw=1.3, label="Q")
        ax.plot(tr.t, tr.R, color="#3CB371", lw=1.3, label="R")
        ax.plot(tr.t, tr.D, color="#222222", lw=1.3, label="D")
        ax.axhline(p.I_cap, color="#DC3232", lw=0.9, ls=":", alpha=0.7)
        ax.text(tr.t[-1], p.I_cap, " I_cap", color="#DC3232",
                fontsize=6, va="bottom", ha="right")
        ax.axvline(t_days, color="#00CC66", lw=1.0, alpha=0.8)
        ax.set_ylim(0, 1)
        ax.set_xlim(tr.t[0], tr.t[-1])
        ax.set_xlabel("t (days)", fontsize=7)
        ax.set_ylabel("fraction", fontsize=7)
        ax.set_title("Optimal epidemic trajectory (ODE = ground truth)",
                     fontsize=8)
        ax.legend(fontsize=6, ncol=5, loc="upper right", columnspacing=0.8,
                  handlelength=1.1)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.15)

    def _draw_controls(self, t_days):
        ax, axr = self.ax_ctrl, self.ax_reff
        ax.clear()
        axr.clear()
        tr = self.trajectory
        ax.fill_between(tr.t, 0, tr.u_L, color="#4C72B0", alpha=0.30)
        ax.plot(tr.t, tr.u_L, color="#4C72B0", lw=1.5, label="u_L lockdown")
        ax.plot(tr.t, tr.u_Q, color="#FFA500", lw=1.5, label="u_Q isolation")
        ax.axvline(t_days, color="#00CC66", lw=1.0, alpha=0.8)
        ax.set_ylim(0, 1)
        ax.set_xlim(tr.t[0], tr.t[-1])
        ax.set_xlabel("t (days)", fontsize=7)
        ax.set_ylabel("control", fontsize=7)
        ax.set_title("Optimal controls  &  effective reproduction number",
                     fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.15)

        axr.plot(tr.t, tr.Reff, color="#8855CC", lw=1.2, ls="--",
                 label="R_eff")
        axr.axhline(1.0, color="#8855CC", lw=0.8, ls=":", alpha=0.6)
        axr.set_ylim(0, max(3.0, float(np.nanmax(tr.Reff)) * 1.1))
        axr.set_ylabel("R_eff", fontsize=7, color="#8855CC")
        axr.tick_params(labelsize=6, colors="#8855CC")

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = axr.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="upper right")

    def _draw_stats(self, t_days):
        ax = self.ax_stats
        ax.clear()
        ax.axis("off")
        tr, p = self.trajectory, self.params
        smp = tr.at(t_days)
        cb = cost_breakdown(tr)

        col1 = [
            "── now ──",
            f"t      = {t_days:6.1f} d",
            f"S      = {smp['S']:.3f}",
            f"I      = {smp['I']:.3f}",
            f"Q      = {smp['Q']:.3f}",
            f"R      = {smp['R']:.3f}",
            f"D      = {smp['D']:.4f}",
            f"u_L    = {smp['u_L']:.2f}",
            f"u_Q    = {smp['u_Q']:.2f}",
            f"R_eff  = {smp['Reff']:.2f}",
        ]
        col2 = [
            "── scenario ──",
            f"R0      = {r0(p):.2f}",
            f"R0 fit  = {self.info.get('R0_fitted', float('nan')):.2f}",
            f"peak I  = {tr.I.max():.3f}",
            f"cap     = {p.I_cap:.3f}",
            f"finalD  = {tr.D[-1]*100:.2f}%",
            "── cost J ──",
            f"burden  = {cb['burden']:.3f}",
            f"deaths  = {cb['deaths']:.3f}",
            f"lockdwn = {cb['lockdown']:.3f}",
            f"quaran. = {cb['quarantine']:.3f}",
            f"capacty = {cb['capacity']:.3f}",
            f"TOTAL J = {cb['total']:.3f}",
        ]
        foot = (f"solve {self._last_solve_ms:.0f}ms · {self._worker_status}"
                f" · {'conv' if tr.converged else 'partial'} · fps {self.fps:.0f}")

        for i, line in enumerate(col1):
            c = "#8888aa" if line.startswith("──") else "#cccccc"
            ax.text(0.02, 0.97 - i * 0.085, line, fontsize=6.5,
                    family="monospace", color=c, va="top", transform=ax.transAxes)
        for i, line in enumerate(col2):
            c = "#8888aa" if line.startswith("──") else "#cccccc"
            ax.text(0.52, 0.97 - i * 0.072, line, fontsize=6.5,
                    family="monospace", color=c, va="top", transform=ax.transAxes)
        ax.text(0.02, 0.02, foot, fontsize=6, family="monospace",
                color="#777799", va="bottom", transform=ax.transAxes)

    def shutdown(self):
        self.req_q.put(None)
