"""
Dashboard with two panels rendered via Agg into a pygame surface.

Panel 1: Population trajectory — S (blue), I (red), R (green), D (black)
Panel 2: HJB Free Boundary (stopping region)

Worker thread re-solves HJB when alpha or ABM-derived rates change.
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

from models.adapter import sirds_euler, solve_hjb

REFRESH_INTERVAL = 0.5


class Dashboard:

    def __init__(self, beta, gamma, mu, V, u_opt, g_array, S_grid, I_grid,
                 alpha_i, alpha_d, i_cap, w_h, T, stopped_frac=0.0):
        self.beta = beta
        self.gamma = gamma
        self.mu = mu
        self.omega = 0.0
        self.V = V
        self.u_opt = u_opt
        self.g_array = g_array
        self.S_grid = S_grid
        self.I_grid = I_grid
        self.alpha_i = alpha_i
        self.alpha_d = alpha_d
        self.i_cap = i_cap
        self.w_h = w_h
        self.T = T
        self.stopping = stopped_frac > 0.01
        self.I0 = 5 / 8000.0
        self.S0 = 1.0 - self.I0

        self.beta_eff = beta
        self.gamma_eff = gamma

        self.paused = False
        self._phase_S = []
        self._phase_I = []

        self._last_solve_dur = 0.0
        self._worker_status = "idle"
        self._fps = 0.0
        self._frame_count = 0
        self._fps_timer = time.monotonic()

        self.solve_request_q = queue.Queue(maxsize=1)
        self.solve_result_q = queue.Queue()
        self.worker = threading.Thread(target=self._solver_loop, daemon=True)
        self.worker.start()

        self.fig = plt.figure(figsize=(5.5, 7.0), dpi=100)
        gs = self.fig.add_gridspec(
            2, 2, width_ratios=[1, 0.45],
            hspace=0.4, wspace=0.35,
            left=0.10, right=0.97, top=0.95, bottom=0.06,
        )
        self.ax_traj = self.fig.add_subplot(gs[0, 0])
        self.ax_stop = self.fig.add_subplot(gs[1, 0])
        self.ax_debug = self.fig.add_subplot(gs[:, 1])
        self.ax_debug.axis("off")

        self.canvas = FigureCanvasAgg(self.fig)
        self.surface = None
        self._last_render = 0.0

    # --- worker (only solve_hjb) --------------------------------------------

    def _solver_loop(self):
        while True:
            req = self.solve_request_q.get()
            if req is None:
                return
            alpha_i, alpha_d, beta_eff, gamma_eff, mu, i_cap, w_h = req
            self._worker_status = f"solving αI={alpha_i:.1f} αD={alpha_d:.0f}"
            t0 = time.monotonic()
            try:
                V_new, u_new, S_g, I_g, g_new, stopped_frac = solve_hjb(
                    beta_eff, gamma_eff, alpha_i, alpha_d=alpha_d,
                    T=self.T, mu=mu, i_cap=i_cap, w_h=w_h,
                )
                dur = time.monotonic() - t0
                self.solve_result_q.put((alpha_i, alpha_d, beta_eff, gamma_eff,
                                         V_new, u_new, S_g, I_g, g_new,
                                         stopped_frac, dur))
            except Exception as e:
                self.solve_result_q.put(("error", e))
            self._worker_status = "idle"

    def request_solve(self, alpha_i, alpha_d, beta_eff, gamma_eff, mu,
                      i_cap, w_h):
        try:
            self.solve_request_q.get_nowait()
        except queue.Empty:
            pass
        self.solve_request_q.put_nowait((alpha_i, alpha_d, beta_eff, gamma_eff,
                                         mu, i_cap, w_h))

    def poll_worker(self):
        try:
            result = self.solve_result_q.get_nowait()
            if result[0] == "error":
                self._worker_status = f"error: {result[1]}"
                return
            (alpha_i_done, alpha_d_done, beta_eff, gamma_eff,
             V_new, u_new, S_g, I_g, g_new,
             stopped_frac, dur) = result
            self.alpha_i = alpha_i_done
            self.alpha_d = alpha_d_done
            self.beta_eff = beta_eff
            self.gamma_eff = gamma_eff
            self.V = V_new
            self.u_opt = u_new
            self.S_grid = S_g
            self.I_grid = I_g
            self.g_array = g_new
            self.stopping = stopped_frac > 0.01
            self._last_solve_dur = dur
        except queue.Empty:
            pass

    # --- tracking -----------------------------------------------------------

    def accumulate(self, t, S, I):
        self._phase_S.append(S)
        self._phase_I.append(I)

    def reset_tracking(self):
        self._phase_S.clear()
        self._phase_I.clear()

    # --- render -------------------------------------------------------------

    def render_to_surface(self, shared_state):
        now = time.monotonic()
        if now - self._last_render < REFRESH_INTERVAL:
            return self.surface
        self._last_render = now

        self._frame_count += 1
        elapsed = now - self._fps_timer
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_timer = now

        self.poll_worker()
        self._draw_trajectory(self.ax_traj, shared_state)
        self._draw_free_boundary(self.ax_stop)
        self._draw_debug(self.ax_debug, shared_state)

        self.canvas.draw()
        buf = self.canvas.buffer_rgba()
        w, h = self.canvas.get_width_height()
        rgb = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()
        self.surface = pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")
        return self.surface

    # --- Panel 1: Population S/I/R/D ---------------------------------------

    def _draw_trajectory(self, ax, state):
        ax.clear()
        with state["lock"]:
            t = list(state["t_history"])
            S = list(state["S_history"])
            I = list(state["I_history"])
            R = list(state["R_history"])
            D = list(state.get("D_history", []))

        if len(t) < 2:
            ax.set_title("Population (waiting)", fontsize=8)
            ax.tick_params(labelsize=6)
            return

        t_arr = np.array(t)
        n = min(len(t_arr), len(S), len(I), len(R))
        ax.plot(t_arr[:n], S[:n], "b-", lw=1.0, label="S")
        ax.plot(t_arr[:n], I[:n], "r-", lw=1.0, label="I")
        ax.plot(t_arr[:n], R[:n], "g-", lw=1.0, label="R")
        if D:
            nd = min(n, len(D))
            ax.plot(t_arr[:nd], D[:nd], "k-", lw=1.2, label="D")

        if self.beta_eff > 0 and self.gamma_eff > 0 and len(t) > 20:
            n_pts = min(200, len(t))
            S_f, I_f, R_f, D_f, t_f = sirds_euler(
                self.beta_eff, self.gamma_eff, self.mu,
                self.S0, self.I0, t_arr[-1], n_pts, omega=self.omega)
            ax.plot(t_f, S_f, "b--", lw=0.5, alpha=0.4)
            ax.plot(t_f, I_f, "r--", lw=0.5, alpha=0.4)
            ax.plot(t_f, R_f, "g--", lw=0.5, alpha=0.4)
            ax.plot(t_f, D_f, "k--", lw=0.5, alpha=0.4)

        ax.axvline(0, color='#00CC66', lw=0.8, ls='--', alpha=0.5)

        ax.set_ylim(0, 1)
        ax.set_xlabel("t (days)", fontsize=7)
        ax.set_ylabel("fraction", fontsize=7)
        ax.set_title(f"SIRD-S  " + r"$\beta$" + f"*={self.beta_eff:.3f}  "
                     + r"$\gamma$" + f"*={self.gamma_eff:.3f}  "
                     + r"$\mu$" + f"={self.mu:.4f}"
                     + f"  I0={self.I0:.4f}", fontsize=7)
        ax.legend(fontsize=5, loc="right")
        ax.tick_params(labelsize=6)

    # --- Panel 2: HJB Free Boundary ----------------------------------------

    def _draw_free_boundary(self, ax):
        ax.clear()
        if self.V is None or self.g_array is None:
            ax.set_title("Free boundary (waiting)", fontsize=8)
            ax.tick_params(labelsize=6)
            return

        stop_mask = np.isclose(self.V, self.g_array, rtol=1e-3, atol=1e-5)
        nS, nI = self.V.shape
        s_c = np.linspace(0, 1, nS)
        i_c = np.linspace(0, 1, nI)

        ax.contourf(i_c, s_c, stop_mask.astype(float),
                    levels=[0.5, 1.5], colors=["#ffcccc"], alpha=0.5)
        ax.contour(i_c, s_c, stop_mask.astype(float),
                   levels=[0.5], colors=["#cc0000"], linewidths=1.0)

        ax.plot(self.I0, self.S0, marker='*', color='#00CC66',
                ms=10, mec='white', mew=0.5, zorder=5)

        if len(self._phase_S) > 1:
            ax.plot(self._phase_I, self._phase_S, "k-", lw=0.5, alpha=0.4)
            ax.plot(self._phase_I[-1], self._phase_S[-1], "ro", ms=5)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("I", fontsize=7)
        ax.set_ylabel("S", fontsize=7)
        ax.set_title(f"HJB Free Boundary  αI={self.alpha_i:.1f} αD={self.alpha_d:.0f}", fontsize=8)
        ax.tick_params(labelsize=6)

    # --- Debug panel --------------------------------------------------------

    def _draw_debug(self, ax, state):
        ax.clear()
        ax.axis("off")
        with state["lock"]:
            t = state.get("t_days", 0.0)
            u = state.get("u_current", 0.0)
            S_h = list(state.get("S_history", []))
            I_h = list(state.get("I_history", []))
            R_h = list(state.get("R_history", []))
            D_h = list(state.get("D_history", []))
            omega = state.get("omega", 0.0)

        S = S_h[-1] if S_h else 0.0
        I = I_h[-1] if I_h else 0.0
        R = R_h[-1] if R_h else 0.0
        D = D_h[-1] if D_h else 0.0
        R0eff = self.beta_eff * (1 - u) * S / self.gamma_eff if self.gamma_eff > 0 else 0.0
        cfr = self.mu / (self.mu + self.gamma) if (self.mu + self.gamma) > 0 else 0.0

        inf_r = state.get("infection_radius", 4)
        jump_p = state.get("jump_prob", 0.007)
        q_delay = state.get("quarantine_delay", 3.0)

        stopping_str = "Yes" if self.stopping else "No"
        lines = [
            "── State ──",
            f"t     = {t:.1f}",
            f"S     = {S:.4f}",
            f"I     = {I:.4f}",
            f"R     = {R:.4f}",
            f"D     = {D:.4f}",
            f"u*    = {u:.4f}",
            f"R0eff = {R0eff:.2f}",
            f"Stopping: {stopping_str}",
            "",
            "── Cost ──",
            f"αI    = {self.alpha_i:.2f}",
            f"αD    = {self.alpha_d:.0f}",
            f"I_cap = {self.i_cap:.3f}",
            f"w_h   = {self.w_h:.0f}",
            f"I0    = {self.I0:.4f}",
            "",
            "── Params ──",
            f"β     = {self.beta:.4f}",
            f"γ     = {self.gamma:.4f}",
            f"mu    = {self.mu:.5f}",
            f"CFR   = {cfr*100:.2f}%",
            f"ω     = {omega:.4f}",
            "",
            "── ABM ──",
            f"inf_r = {inf_r}",
            f"jump  = {jump_p:.4f}",
            f"q_del = {q_delay:.1f}",
            f"β_eff = {self.beta_eff:.4f}",
            f"γ_eff = {self.gamma_eff:.4f}",
            "",
            "── Worker ──",
            f"{self._worker_status}",
            f"last: {self._last_solve_dur:.1f}s",
            f"fps:  {self._fps:.0f}",
        ]
        for idx, line in enumerate(lines):
            color = "#cccccc"
            if line.startswith("──"):
                color = "#8888aa"
            ax.text(0.05, 0.97 - idx * 0.033, line, fontsize=5,
                    family="monospace", color=color, va="top",
                    transform=ax.transAxes)
