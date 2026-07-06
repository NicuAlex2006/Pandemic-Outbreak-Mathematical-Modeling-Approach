"""
Renderer — pygame view of the agent population (which renders the optimal ODE)
plus parameter sliders that re-solve the optimal control on the fly.
"""

import time
from enum import Enum

import numpy as np
import pygame

from simulation.agent import (
    Agents, N_COMMUNITIES, COMMUNITY_SIZE, QUARANTINE_RECT,
)
from models.adapter import with_overrides

SIM_W = 900
DASH_W = 560
WINDOW_W = SIM_W + DASH_W
WINDOW_H = 800
FIELD_H = 590
TARGET_FPS = 30

COLORS = {
    "bg":        (18, 18, 28),
    "panel":     (26, 26, 38),
    "community": (52, 52, 72),
    "zone":      (170, 110, 40),
    "text":      (220, 220, 225),
    "dim":       (130, 130, 150),
    "accent":    (0, 204, 102),
}
STATUS_RGB = np.array([
    (100, 149, 237),   # S
    (220, 50, 50),     # I
    (255, 165, 0),     # Q
    (60, 179, 113),    # R
    (70, 70, 78),      # D
], dtype=np.uint8)


class GameState(Enum):
    PLAYING = "playing"
    PAUSED = "paused"


class Slider:
    def __init__(self, x, y, w, label, lo, hi, val, fmt=".2f", log=False):
        self.x, self.y, self.w = x, y, w
        self.label, self.lo, self.hi = label, float(lo), float(hi)
        self.value = float(val)
        self.fmt, self.log = fmt, log
        self.dragging = False
        self.r = 6

    def _frac(self):
        if self.log:
            return (np.log(self.value) - np.log(self.lo)) / (np.log(self.hi) - np.log(self.lo))
        return (self.value - self.lo) / (self.hi - self.lo)

    def _hx(self):
        return int(self.x + np.clip(self._frac(), 0, 1) * self.w)

    def draw(self, surf, font):
        pygame.draw.rect(surf, (70, 70, 95), (self.x, self.y + 12, self.w, 3))
        pygame.draw.circle(surf, (160, 160, 215), (self._hx(), self.y + 13), self.r)
        txt = font.render(f"{self.label}: {self.value:{self.fmt}}", True, COLORS["dim"])
        surf.blit(txt, (self.x, self.y - 3))

    def handle(self, event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            hx = self._hx()
            if (event.pos[0] - hx) ** 2 + (event.pos[1] - self.y - 13) ** 2 <= (self.r + 6) ** 2:
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            f = float(np.clip((event.pos[0] - self.x) / self.w, 0, 1))
            if self.log:
                self.value = float(np.exp(np.log(self.lo) + f * (np.log(self.hi) - np.log(self.lo))))
            else:
                self.value = self.lo + f * (self.hi - self.lo)
            return True
        return False


class Button:
    def __init__(self, x, y, w, h, label, color, hover, border):
        self.rect = pygame.Rect(x, y, w, h)
        self.label, self.color, self.hover, self.border = label, color, hover, border
        self._hov = False

    def draw(self, surf, font):
        pygame.draw.rect(surf, self.hover if self._hov else self.color, self.rect, border_radius=4)
        pygame.draw.rect(surf, self.border, self.rect, 1, border_radius=4)
        txt = font.render(self.label, True, (225, 225, 225))
        surf.blit(txt, txt.get_rect(center=self.rect.center))

    def handle(self, event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self._hov = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(event.pos)
        return False


class Renderer:
    def __init__(self, world, dashboard, base_params):
        self.world = world
        self.dashboard = dashboard
        self.base_params = base_params
        self.state = GameState.PAUSED
        self.running = False

        self._fps = 0.0
        self._fps_t = time.monotonic()
        self._frames = 0
        self._resolve_at = 0.0
        self._restart_pending = False

        p = base_params
        sw, x0 = 120, 20
        self.s_beta = Slider(x0, 620, sw, "beta", 0.05, 1.0, p.beta, ".2f")
        self.s_gamma = Slider(x0 + 170, 620, sw, "gamma", 0.03, 0.4, p.gamma, ".3f")
        self.s_mu = Slider(x0 + 340, 620, sw, "mu", 0.0, 0.02, p.mu, ".4f")

        self.s_icap = Slider(x0, 685, sw, "I_cap", 0.02, 0.2, p.I_cap, ".3f")
        self.s_cL = Slider(x0 + 170, 685, sw, "c_L lockdown$", 0.1, 3.0, p.c_L, ".2f")
        self.s_uQ = Slider(x0 + 340, 685, sw, "isolation cap", 0.0, 0.3, p.u_Q_max, ".3f")
        self.s_wI = Slider(x0 + 510, 685, sw, "w_I burden", 0.0, 0.2, p.w_I, ".3f")

        self.s_speed = Slider(x0, 748, sw, "speed d/s", 2, 40, 15, ".0f")
        self.s_init = Slider(x0 + 170, 748, sw, "init I", 1, 50, 8, ".0f")

        self.model_sliders = [self.s_beta, self.s_gamma, self.s_mu,
                              self.s_icap, self.s_cL, self.s_uQ, self.s_wI]
        self.all_sliders = self.model_sliders + [self.s_speed, self.s_init]

        self.btn_play = Button(700, 612, 180, 30, "PLAY",
                               (25, 60, 35), (40, 100, 55), (60, 180, 90))
        self.btn_restart = Button(700, 650, 180, 30, "RESTART",
                                  (70, 35, 35), (120, 55, 55), (180, 70, 70))

    # --- setup ---------------------------------------------------------------

    def setup(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Optimal Epidemic Control — SIRDQ")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 13)
        self.small = pygame.font.SysFont("consolas", 11)
        self.bold = pygame.font.SysFont("consolas", 13, bold=True)
        self.field_surf = pygame.Surface((SIM_W, FIELD_H))
        self.field = np.empty((SIM_W, FIELD_H, 3), dtype=np.uint8)

    # --- params / solve ------------------------------------------------------

    def _current_params(self):
        return with_overrides(
            self.base_params,
            beta=self.s_beta.value, gamma=self.s_gamma.value, mu=self.s_mu.value,
            I_cap=self.s_icap.value, c_L=self.s_cL.value,
            u_Q_max=self.s_uQ.value, w_I=self.s_wI.value)

    def _request_resolve(self):
        self.dashboard.request_solve(self._current_params())

    # --- main loop -----------------------------------------------------------

    def run(self):
        self.running = True
        self._request_resolve()
        while self.running:
            dt_real = self.clock.tick(TARGET_FPS) / 1000.0
            self._events()
            self._check_resolve()

            new = self.dashboard.poll_worker()
            if new is not None:
                self.world.set_trajectory(new)

            if self.state == GameState.PLAYING and self.world.trajectory is not None:
                self.world.step(self.s_speed.value * dt_real)
                if self.world.finished():
                    self._set_state(GameState.PAUSED)

            self._update_fps()
            self.draw()
            pygame.display.flip()

        self.dashboard.shutdown()
        pygame.quit()

    def _set_state(self, st):
        self.state = st
        if st == GameState.PLAYING:
            self.btn_play.label = "PAUSE"
            self.btn_play.color, self.btn_play.hover, self.btn_play.border = \
                (55, 55, 25), (90, 90, 40), (180, 180, 60)
        else:
            self.btn_play.label = "REPLAY" if self.world.finished() else "PLAY"
            self.btn_play.color, self.btn_play.hover, self.btn_play.border = \
                (25, 60, 35), (40, 100, 55), (60, 180, 90)

    def _check_resolve(self):
        if self._resolve_at and time.monotonic() >= self._resolve_at:
            self._resolve_at = 0.0
            self._request_resolve()

    def _update_fps(self):
        self._frames += 1
        now = time.monotonic()
        if now - self._fps_t >= 1.0:
            self._fps = self._frames / (now - self._fps_t)
            self._frames = 0
            self._fps_t = now
            self.dashboard.fps = self._fps

    # --- events --------------------------------------------------------------

    def _events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    self.running = False
                elif e.key == pygame.K_SPACE:
                    self._toggle()
                elif e.key == pygame.K_r:
                    self._restart()
            if self.btn_play.handle(e):
                self._toggle()
            if self.btn_restart.handle(e):
                self._restart()
            for s in self.all_sliders:
                if s.handle(e):
                    self._on_slider(s)

    def _toggle(self):
        if self.state == GameState.PLAYING:
            self._set_state(GameState.PAUSED)
        else:
            if self.world.finished():
                self._restart()
            self._set_state(GameState.PLAYING)

    def _restart(self):
        self.world.reset(int(self.s_init.value))
        self._restart_pending = False
        self._set_state(GameState.PAUSED)

    def _on_slider(self, s):
        if s is self.s_init:
            self._restart_pending = True
            return
        if s in self.model_sliders:
            # debounce: re-solve shortly after the user stops dragging
            self._resolve_at = time.monotonic() + 0.12

    # --- drawing -------------------------------------------------------------

    def draw(self):
        self._draw_field()
        self._draw_hud()
        self._draw_panel()
        surf = self.dashboard.render_to_surface(self.world.t_days)
        if surf:
            self.screen.blit(surf, (SIM_W, 0))

    def _draw_field(self):
        a = self.world.agents
        self.field[:] = COLORS["bg"]
        if a.positions is not None:
            xs = a.positions[:, 0].astype(np.intp)
            ys = a.positions[:, 1].astype(np.intp)
            cols = STATUS_RGB[a.statuses]
            for dx in (0, 1):
                for dy in (0, 1):
                    xx, yy = xs + dx, ys + dy
                    v = (xx >= 0) & (xx < SIM_W) & (yy >= 0) & (yy < FIELD_H)
                    self.field[xx[v], yy[v]] = cols[v]
        pygame.surfarray.blit_array(self.field_surf, self.field)
        self.screen.blit(self.field_surf, (0, 0))

        # community rects + quarantine zone on top (thin outlines)
        if self.world.centers is not None:
            for cx, cy in self.world.centers.astype(int):
                pygame.draw.rect(self.screen, COLORS["community"],
                                 (cx, cy, COMMUNITY_SIZE[0], COMMUNITY_SIZE[1]), 1)
        qx, qy, qw, qh = QUARANTINE_RECT
        pygame.draw.rect(self.screen, COLORS["zone"], (qx, qy, qw, qh), 1)
        self.screen.blit(self.small.render("QUARANTINE", True, COLORS["zone"]),
                         (qx + 6, qy + 5))

    def _draw_hud(self):
        a = self.world.agents
        nS, nI, nQ, nR, nD = a.counts()
        smp = (self.world.trajectory.at(self.world.t_days)
               if self.world.trajectory is not None else None)
        lines = [
            (f"Day {self.world.t_days:6.1f}", COLORS["text"]),
            (f"u_L {self.world.u_L:.2f}   u_Q {self.world.u_Q:.2f}", (130, 200, 255)),
        ]
        if smp:
            lines.append((f"R_eff {smp['Reff']:.2f}", (180, 140, 220)))
        lines += [
            (f"I {nI}  Q {nQ}", (235, 130, 90)),
            (f"S {nS}  R {nR}  D {nD}", COLORS["dim"]),
        ]
        for i, (t, c) in enumerate(lines):
            self.screen.blit(self.font.render(t, True, c), (10, 8 + i * 17))

        tag = {GameState.PLAYING: ("RUNNING", (100, 200, 255)),
               GameState.PAUSED: ("PAUSED", (255, 230, 120))}[self.state]
        self.screen.blit(self.font.render(tag[0], True, tag[1]), (700, 590))

    def _draw_panel(self):
        pygame.draw.rect(self.screen, COLORS["panel"], (0, 600, SIM_W, WINDOW_H - 600))
        pygame.draw.line(self.screen, (200, 80, 80), (12, 612), (340, 612), 1)
        self.screen.blit(self.bold.render("THE VIRUS", True, (210, 90, 90)), (12, 600))
        pygame.draw.line(self.screen, (80, 130, 200), (12, 677), (520, 677), 1)
        self.screen.blit(self.bold.render("THE POLICY  (cost knobs → optimal control)",
                                          True, (90, 140, 210)), (12, 665))
        self.screen.blit(self.small.render("Simulation", True, COLORS["dim"]), (12, 730))
        for s in self.all_sliders:
            s.draw(self.screen, self.small)
        if self._restart_pending:
            self.btn_restart.label = "RESTART *"
        self.btn_play.draw(self.screen, self.small)
        self.btn_restart.draw(self.screen, self.small)
