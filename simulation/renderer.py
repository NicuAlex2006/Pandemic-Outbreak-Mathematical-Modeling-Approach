import math
import time
from enum import Enum
import pygame
import numpy as np

import simulation.agent as _agent
from simulation.agent import Agents, COMMUNITY_SIZE, QUARANTINE_RECT

SIM_W = 900
DASHBOARD_W = 550
WINDOW_W = SIM_W + DASHBOARD_W
WINDOW_H = 780
TARGET_FPS = 30

COLORS = {
    'bg':         (20,  20,  30),
    'S':          (100, 149, 237),
    'I':          (220,  50,  50),
    'R':          (60,  179, 113),
    'Q':          (255, 165,   0),
    'D':          (40,  40,  40),
    'community':  (55,   55,  75),
    'text':       (220, 220, 220),
    'dim':        (130, 130, 150),
    'quarantine': (180, 100,  30),
    'computing':  (255, 200,  80),
    'virus_accent':  (180, 60, 60),
    'battle_accent': (60, 130, 190),
    'locked_track':  (50, 50, 60),
    'locked_handle': (75, 75, 85),
    'locked_text':   (80, 80, 95),
}

_STATUS_COLORS = [COLORS['S'], COLORS['I'], COLORS['R'], COLORS['Q'], COLORS['D']]

BASE_INFECTION_RADIUS = _agent.INFECTION_RADIUS
BASE_JUMP_PROB = _agent.JUMP_PROB
BASE_QUARANTINE_DELAY = _agent.QUARANTINE_DELAY

# ────────────────────────────────────────────────────────────────────────────
# Policy-distribution model (dual to the HJB).
#
# The HJB returns a scalar u ∈ [0, 1] — the optimal *total* intervention
# intensity. We must split it across three concrete policy levers without
# letting any single lever collapse to its extreme by default. Each lever has:
#
#   w_i : effectiveness  (units of transmission suppression per unit restriction)
#   c_i : societal cost  (quadratic — marginal pain grows with restriction)
#
# We solve the convex QP
#     min  Σ c_i · r_i²
#     s.t. Σ w_i · r_i  =  u · Σ w_i,    r_i ∈ [0, 1]
#
# KKT optimum (unsaturated):  r_i = λ · w_i / c_i,  λ = u·Σw_i / Σ(w_i²/c_i)
# Saturation: when r_i > 1 we clip, subtract its supplied suppression from the
# RHS, and resolve over the remaining active levers (active-set method).
#
# Tuned so the qualitative ordering matches policy intuition:
#   cheap → expensive  :  jump  <  radius  <  q_delay
#   effectiveness      :  radius >  jump   ≈ q_delay
# i.e. mobility restrictions ramp first, lockdown-grade isolation comes later.
# ────────────────────────────────────────────────────────────────────────────
POLICY_LEVERS = (
    # (name,    effectiveness w, cost c, base EMA gain)
    ("jump",    1.0,             0.5,    0.08),
    ("rad",     1.5,             1.0,    0.05),
    ("delay",   1.0,             2.0,    0.03),
)

# Urgency in [0, 1] drives both adaptation speed and policy-enaction probability.
# speed_mult = clip(URGENCY_SPEED_GAIN * urgency, URGENCY_SPEED_MIN, URGENCY_SPEED_MAX)
URGENCY_SPEED_GAIN = 3.0
URGENCY_SPEED_MIN  = 0.15
URGENCY_SPEED_MAX  = 3.0

# Bernoulli probability that a given lever is actually enacted on a given step:
#   p_enact = clip(urgency * care, P_ENACT_MIN, 1.0)
P_ENACT_MIN = 0.02   # even fully apathetic, policies sometimes change


class GameState(Enum):
    SETUP = "setup"
    PLAYING = "playing"
    PAUSED = "paused"


class Slider:
    def __init__(self, x, y, w, label, min_val, max_val, initial, fmt=".2f",
                 category="sim"):
        self.x, self.y, self.w = x, y, w
        self.label = label
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.value = float(initial)
        self.fmt = fmt
        self.dragging = False
        self.handle_r = 7
        self.track = None
        self.category = category
        self.locked = False
        self.ai_controlled = False

    def setup(self):
        self.track = pygame.Rect(self.x, self.y + 12, self.w, 3)

    def _handle_x(self):
        frac = (self.value - self.min_val) / (self.max_val - self.min_val)
        return int(self.x + frac * self.w)

    def draw(self, surface, font):
        hx = self._handle_x()
        if self.ai_controlled:
            pygame.draw.rect(surface, (30, 60, 65), self.track)
            pygame.draw.circle(surface, (80, 200, 210),
                               (hx, self.y + 13), self.handle_r)
            txt = font.render(f'{self.label}: {self.value:{self.fmt}}',
                              True, (80, 200, 210))
            surface.blit(txt, (self.x, self.y - 2))
        elif self.locked:
            pygame.draw.rect(surface, COLORS['locked_track'], self.track)
            pygame.draw.circle(surface, COLORS['locked_handle'],
                               (hx, self.y + 13), self.handle_r)
            txt = font.render(f'{self.label}: {self.value:{self.fmt}}',
                              True, COLORS['locked_text'])
            surface.blit(txt, (self.x, self.y - 2))
            lx = self.x + self.w + 5
            ly = self.y + 4
            pygame.draw.rect(surface, COLORS['locked_text'],
                             (lx + 1, ly, 7, 5), 1, border_radius=2)
            pygame.draw.rect(surface, COLORS['locked_text'],
                             (lx, ly + 4, 9, 7), border_radius=1)
        else:
            pygame.draw.rect(surface, (75, 75, 100), self.track)
            pygame.draw.circle(surface, (160, 160, 210),
                               (hx, self.y + 13), self.handle_r)
            txt = font.render(f'{self.label}: {self.value:{self.fmt}}',
                              True, COLORS['dim'])
            surface.blit(txt, (self.x, self.y - 2))

    def handle_event(self, event) -> bool:
        if self.locked or self.ai_controlled:
            return False
        hx = self._handle_x()
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            dx, dy = event.pos[0] - hx, event.pos[1] - (self.y + 13)
            if dx * dx + dy * dy <= (self.handle_r + 4) ** 2:
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            frac = float(np.clip((event.pos[0] - self.x) / self.w, 0.0, 1.0))
            self.value = self.min_val + frac * (self.max_val - self.min_val)
            return True
        return False


class Button:
    def __init__(self, x, y, w, h, label,
                 color=(70, 35, 35), hover_color=(120, 55, 55),
                 border_color=(180, 70, 70)):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.color = color
        self.hover_color = hover_color
        self.border_color = border_color
        self._hovered = False

    def draw(self, surface, font):
        fill = self.hover_color if self._hovered else self.color
        pygame.draw.rect(surface, fill, self.rect, border_radius=4)
        pygame.draw.rect(surface, self.border_color, self.rect, 1, border_radius=4)
        txt = font.render(self.label, True, (220, 220, 220))
        surface.blit(txt, txt.get_rect(center=self.rect.center))

    def handle_event(self, event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self._hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False


class Renderer:
    def __init__(self, world, shared_state: dict, dashboard=None):
        self.world = world
        self.state = shared_state
        self.dashboard = dashboard
        self.screen = None
        self.clock = None
        self.font = None
        self.font_small = None
        self.font_section = None
        self.running = False
        self.on_step = None
        self.on_restart = None
        self._active_initial_infected = 5
        self._restart_pending = False
        self.game_state = GameState.SETUP
        self._resolve_deadline = 0.0

        fitted_beta = dashboard.beta if dashboard else 0.3
        fitted_gamma = dashboard.gamma if dashboard else 0.14
        fitted_mu = dashboard.mu if dashboard else 0.003

        self.btn_play_pause = Button(
            700, 8, 170, 28, 'PLAY',
            color=(25, 60, 35), hover_color=(40, 100, 55),
            border_color=(60, 180, 90))
        self.btn_restart = Button(700, 42, 170, 28, 'RESTART')

        sw = 105

        # ── THE VIRUS (Biological) ── locked during PLAYING ──
        bio_y = 582
        self.slider_beta = Slider(20, bio_y, sw, 'beta', 0.01, 1.0,
                                  fitted_beta, fmt=".4f", category="bio")
        self.slider_gamma = Slider(150, bio_y, sw, 'gamma', 0.01, 0.5,
                                   fitted_gamma, fmt=".4f", category="bio")
        self.slider_mu = Slider(280, bio_y, sw, 'mu', 0.0, 0.05,
                                fitted_mu, fmt=".4f", category="bio")
        self.slider_omega = Slider(410, bio_y, sw, 'omega', 0.0, 0.1,
                                   0.0, fmt=".4f", category="bio")

        # ── THE BATTLE (Policy) ── active during PLAYING ──
        # Two user-facing knobs replace the old (alpha_I, alpha_D, I_cap, w_h):
        #   care    — how strongly to weight infection cost in the HJB
        #   urgency — how fast & how reliably the AI sliders actually move
        pol_y1 = 642
        pol_y2 = 672
        self.slider_care = Slider(20, pol_y1, sw, 'care', 0.0, 1.0,
                                  0.5, fmt=".2f", category="policy")
        self.slider_urgency = Slider(150, pol_y1, sw, 'urgency', 0.0, 1.0,
                                     0.5, fmt=".2f", category="policy")
        self.slider_inf_radius = Slider(20, pol_y2, sw, 'inf_rad', 1, 15,
                                        BASE_INFECTION_RADIUS, fmt=".0f",
                                        category="policy")
        self.slider_jump_prob = Slider(150, pol_y2, sw, 'jump_p', 0.0, 0.05,
                                       BASE_JUMP_PROB, fmt=".4f",
                                       category="policy")
        self.slider_q_delay = Slider(280, pol_y2, sw, 'q_delay', 0.5, 10.0,
                                     BASE_QUARANTINE_DELAY, fmt=".1f",
                                     category="policy")

        # ── Simulation Controls ──
        ctrl_y = 730
        self.slider_speed = Slider(20, ctrl_y, sw, 'speed', 1.0, 8.0, 2.0,
                                   category="sim")
        self.slider_initial_inf = Slider(150, ctrl_y, sw, 'init_I', 1, 500,
                                         5, fmt=".0f", category="sim")

        self.bio_sliders = [
            self.slider_beta, self.slider_gamma,
            self.slider_mu, self.slider_omega,
        ]
        self.cost_sliders = [
            self.slider_care, self.slider_urgency,
        ]
        self.control_sliders = [
            self.slider_inf_radius, self.slider_jump_prob,
            self.slider_q_delay,
        ]
        self.policy_sliders = self.cost_sliders + self.control_sliders
        self.sim_sliders = [self.slider_speed, self.slider_initial_inf]
        self._all_sliders = (self.bio_sliders + self.policy_sliders
                             + self.sim_sliders)

        self._auto_inf_rad = float(BASE_INFECTION_RADIUS)
        self._auto_jump_prob = float(BASE_JUMP_PROB)
        self._auto_q_delay = float(BASE_QUARANTINE_DELAY)

        # Deterministic RNG for the per-step Bernoulli policy-enaction gate.
        self._policy_rng = np.random.default_rng(42)

    def setup(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption('Epidemic Control — SIRD-S')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('consolas', 13)
        self.font_small = pygame.font.SysFont('consolas', 11)
        self.font_section = pygame.font.SysFont('consolas', 13, bold=True)
        for s in self._all_sliders:
            s.setup()

    # ── State Machine ───────────────────────────────────────────────────────

    def _set_state(self, new_state: GameState):
        old = self.game_state
        self.game_state = new_state

        if new_state == GameState.SETUP:
            for s in self.bio_sliders:
                s.locked = False
            for s in self.control_sliders:
                s.ai_controlled = False
                s.locked = False
            if self.dashboard:
                self.dashboard.paused = True
            self.btn_play_pause.label = 'PLAY'
            self.btn_play_pause.color = (25, 60, 35)
            self.btn_play_pause.hover_color = (40, 100, 55)
            self.btn_play_pause.border_color = (60, 180, 90)

        elif new_state == GameState.PLAYING:
            for s in self.bio_sliders:
                s.locked = True
            for s in self.control_sliders:
                s.ai_controlled = True
            self._auto_inf_rad = self.slider_inf_radius.value
            self._auto_jump_prob = self.slider_jump_prob.value
            self._auto_q_delay = self.slider_q_delay.value
            if self.dashboard:
                self.dashboard.paused = False
            self.btn_play_pause.label = 'PAUSE'
            self.btn_play_pause.color = (55, 55, 25)
            self.btn_play_pause.hover_color = (90, 90, 40)
            self.btn_play_pause.border_color = (180, 180, 60)
            if old == GameState.SETUP:
                self._compute_effective_rates()
                self._trigger_resolve()

        elif new_state == GameState.PAUSED:
            if self.dashboard:
                self.dashboard.paused = True
            self.btn_play_pause.label = 'PLAY'
            self.btn_play_pause.color = (25, 60, 35)
            self.btn_play_pause.hover_color = (40, 100, 55)
            self.btn_play_pause.border_color = (60, 180, 90)

    # ── Effective rates / resolve ───────────────────────────────────────────

    def _compute_effective_rates(self):
        if not self.dashboard:
            return
        self.dashboard.beta = self.slider_beta.value
        self.dashboard.gamma = self.slider_gamma.value
        self.dashboard.mu = self.slider_mu.value
        self.dashboard.omega = self.slider_omega.value

        r_ratio = self.slider_inf_radius.value / max(BASE_INFECTION_RADIUS, 1)
        j_ratio = self.slider_jump_prob.value / max(BASE_JUMP_PROB, 1e-6)
        q_ratio = BASE_QUARANTINE_DELAY / max(self.slider_q_delay.value, 0.1)

        self.dashboard.beta_eff = (self.dashboard.beta * r_ratio
                                   * (0.5 + 0.5 * j_ratio))
        self.dashboard.gamma_eff = self.dashboard.gamma * q_ratio
        self.dashboard.care = self.slider_care.value
        self.dashboard.urgency = self.slider_urgency.value

        with self.state["lock"]:
            self.state["omega"] = self.slider_omega.value
            self.state["mu_fit"] = self.slider_mu.value
            self.state["care"] = self.slider_care.value
            self.state["urgency"] = self.slider_urgency.value

    def _trigger_resolve(self):
        if not self.dashboard:
            return
        self._compute_effective_rates()
        self.dashboard.request_solve(
            self.slider_care.value,
            self.dashboard.beta_eff,
            self.dashboard.gamma_eff,
            self.dashboard.mu,
        )

    def _schedule_resolve(self):
        self._resolve_deadline = time.monotonic() + 0.15

    def _check_pending_resolve(self):
        if (self._resolve_deadline > 0
                and time.monotonic() >= self._resolve_deadline):
            self._resolve_deadline = 0.0
            self._trigger_resolve()

    # ── Main loop ───────────────────────────────────────────────────────────

    def run(self):
        self.running = True
        self._set_state(GameState.SETUP)
        self._compute_effective_rates()
        self._trigger_resolve()
        while self.running:
            self.handle_events()
            self._check_pending_resolve()
            if self.game_state == GameState.PLAYING:
                steps = max(1, int(self.slider_speed.value))
                u = 0.0
                for _ in range(steps):
                    u = self._get_u()
                    self.world.set_u(u)
                    self.world.step()
                    if self.on_step:
                        self.on_step()
                    S, I, R = self.world.get_sir()
                    if self.dashboard:
                        self.dashboard.accumulate(self.world.t_days, S, I)
                self._apply_autonomous_control(u)
            self._sync_hjb_solution()
            self.draw()
            pygame.display.flip()
            self.clock.tick(TARGET_FPS)
        if self.dashboard:
            self.dashboard.solve_request_q.put(None)
        pygame.quit()

    def _get_u(self) -> float:
        with self.state['lock']:
            u_opt = self.state.get('u_opt')
            S_grid = self.state.get('S_grid')
            I_grid = self.state.get('I_grid')
            t_days = self.state.get('t_days', 0.0)
        if u_opt is None:
            return 0.0
        from models.adapter import get_u_at
        n_time = u_opt.shape[0]
        t_idx = min(int(t_days / 30.0 * n_time), n_time - 1)
        S, I, _ = self.world.get_sir()
        return get_u_at(S, I, t_idx, u_opt, S_grid, I_grid)

    # ── Autonomous Control ───────────────────────────────────────────────────

    @staticmethod
    def _distribute_policy(u: float) -> dict:
        """Solve the QP that splits HJB target u across the three policy levers.

        Returns dict name -> r_i in [0, 1] (restriction intensity per lever).
        Cheap, effective levers absorb more; saturated levers redistribute
        residual demand to the remaining unsaturated ones.
        """
        u = max(0.0, min(1.0, float(u)))
        if u == 0.0:
            return {name: 0.0 for name, *_ in POLICY_LEVERS}

        r = {name: 0.0 for name, *_ in POLICY_LEVERS}
        active = {name: (w, c) for name, w, c, _ in POLICY_LEVERS}
        sum_w_total = sum(w for _, w, _, _ in POLICY_LEVERS)
        remaining = u * sum_w_total

        # Active-set: at most len(POLICY_LEVERS) passes.
        for _ in range(len(POLICY_LEVERS)):
            denom = sum((w * w) / c for w, c in active.values())
            if denom <= 0 or not active:
                break
            lam = remaining / denom
            newly_clipped = []
            for name, (w, c) in active.items():
                r[name] = lam * w / c
                if r[name] >= 1.0:
                    newly_clipped.append((name, w))
            if not newly_clipped:
                break
            for name, w in newly_clipped:
                r[name] = 1.0
                remaining -= w
                del active[name]
            if remaining <= 1e-9 or not active:
                break
        return r

    def _read_care_urgency(self):
        """Snapshot care, urgency from dashboard (or defaults if absent)."""
        if self.dashboard is None:
            return 0.5, 0.5
        care    = float(getattr(self.dashboard, "care",    0.5))
        urgency = float(getattr(self.dashboard, "urgency", 0.5))
        return (float(np.clip(care, 0.0, 1.0)),
                float(np.clip(urgency, 0.0, 1.0)))

    def _urgency_speed(self, urgency: float) -> float:
        """EMA-speed multiplier from urgency in [0, 1]."""
        return float(np.clip(URGENCY_SPEED_GAIN * urgency,
                             URGENCY_SPEED_MIN, URGENCY_SPEED_MAX))

    def _apply_autonomous_control(self, u):
        # 1. Cost-aware allocation of total intervention u across levers.
        r = self._distribute_policy(u)

        # 2. Map r_i in [0, 1] -> slider value.
        #    r = 0  →  slider at MAX (least restrictive)
        #    r = 1  →  slider at MIN (most restrictive)
        def target_for(slider, r_i):
            return (1.0 - r_i) * slider.max_val + r_i * slider.min_val

        target_jump  = target_for(self.slider_jump_prob,  r["jump"])
        target_rad   = target_for(self.slider_inf_radius, r["rad"])
        target_delay = target_for(self.slider_q_delay,    r["delay"])

        # 3. Adaptation speed: per-lever base gain * speed(urgency).
        care, urgency = self._read_care_urgency()
        speed = self._urgency_speed(urgency)
        base = {name: k for name, _w, _c, k in POLICY_LEVERS}
        k_jump  = base["jump"]  * speed
        k_rad   = base["rad"]   * speed
        k_delay = base["delay"] * speed

        # 4. Probabilistic policy enaction.
        #    Each lever independently draws a Bernoulli per step.
        #    p_enact = clip(urgency * care, P_ENACT_MIN, 1.0)
        #    Low urgency * low care -> sliders rarely move (policy inertia).
        p_enact = float(np.clip(urgency * care, P_ENACT_MIN, 1.0))
        rng = self._policy_rng

        if rng.random() < p_enact:
            self._auto_jump_prob += k_jump * (target_jump - self._auto_jump_prob)
        if rng.random() < p_enact:
            self._auto_inf_rad   += k_rad   * (target_rad   - self._auto_inf_rad)
        if rng.random() < p_enact:
            self._auto_q_delay   += k_delay * (target_delay - self._auto_q_delay)

        self.slider_jump_prob.value = self._auto_jump_prob
        self.slider_inf_radius.value = self._auto_inf_rad
        self.slider_q_delay.value = self._auto_q_delay

        _agent.JUMP_PROB = self._auto_jump_prob
        _agent.INFECTION_RADIUS = max(1, int(round(self._auto_inf_rad)))
        _agent.QUARANTINE_DELAY = self._auto_q_delay

        with self.state["lock"]:
            self.state["jump_prob"] = self._auto_jump_prob
            self.state["infection_radius"] = max(1, int(round(self._auto_inf_rad)))
            self.state["quarantine_delay"] = self._auto_q_delay
            self.state["p_enact"] = p_enact

        self._compute_effective_rates()

    def _sync_hjb_solution(self):
        if not self.dashboard:
            return
        u_opt = self.dashboard.u_opt
        if u_opt is None:
            return
        with self.state["lock"]:
            self.state["u_opt"] = u_opt
            self.state["S_grid"] = self.dashboard.S_grid
            self.state["I_grid"] = self.dashboard.I_grid

    # ── Drawing ─────────────────────────────────────────────────────────────

    def draw(self):
        self.screen.fill(COLORS['bg'])
        self.draw_quarantine_zone()
        self.draw_communities()
        self.draw_agents()
        self.draw_hud()
        self._draw_slider_sections()
        self.draw_sliders()
        self._draw_buttons()
        if self.dashboard:
            surf = self.dashboard.render_to_surface(self.state)
            if surf:
                self.screen.blit(surf, (SIM_W, 0))

    def _draw_slider_sections(self):
        virus_hdr_y = 562
        # Virus section background
        pygame.draw.rect(self.screen, (35, 25, 28),
                         (10, virus_hdr_y - 2, 530, 42))
        pygame.draw.rect(self.screen, COLORS['virus_accent'],
                         (10, virus_hdr_y, 530, 2))
        hdr_col = (COLORS['locked_text'] if self.game_state != GameState.SETUP
                   else COLORS['virus_accent'])
        txt = self.font_section.render('THE VIRUS  [Biological]', True,
                                       hdr_col)
        self.screen.blit(txt, (14, virus_hdr_y + 4))
        if self.game_state != GameState.SETUP:
            lock_lbl = self.font_small.render('LOCKED', True,
                                              COLORS['locked_text'])
            self.screen.blit(lock_lbl, (480, virus_hdr_y + 5))

        battle_hdr_y = 622
        pygame.draw.rect(self.screen, (25, 28, 38),
                         (10, battle_hdr_y - 2, 530, 72))
        pygame.draw.rect(self.screen, COLORS['battle_accent'],
                         (10, battle_hdr_y, 530, 2))
        txt = self.font_section.render('THE BATTLE  [Policy]', True,
                                       COLORS['battle_accent'])
        self.screen.blit(txt, (14, battle_hdr_y + 4))
        if self.game_state == GameState.PLAYING:
            ai_lbl = self.font_small.render('AI CONTROL', True, (80, 200, 210))
            self.screen.blit(ai_lbl, (460, battle_hdr_y + 5))

        ctrl_hdr_y = 712
        pygame.draw.rect(self.screen, COLORS['dim'],
                         (10, ctrl_hdr_y, 280, 1))
        txt = self.font_small.render('Controls', True, COLORS['dim'])
        self.screen.blit(txt, (14, ctrl_hdr_y + 3))

    def _draw_buttons(self):
        if self._restart_pending:
            self.btn_restart.color = (120, 100, 20)
            self.btn_restart.hover_color = (180, 150, 30)
            self.btn_restart.border_color = (255, 200, 50)
            self.btn_restart.label = 'RESTART *'
        else:
            self.btn_restart.color = (70, 35, 35)
            self.btn_restart.hover_color = (120, 55, 55)
            self.btn_restart.border_color = (180, 70, 70)
            self.btn_restart.label = 'RESTART'

        state_labels = {
            GameState.SETUP:   ('SETUP',   (100, 180, 100)),
            GameState.PLAYING: ('RUNNING', (100, 200, 255)),
            GameState.PAUSED:  ('PAUSED',  (255, 255, 100)),
        }
        label, color = state_labels[self.game_state]
        txt = self.font.render(label, True, color)
        self.screen.blit(txt, (700, 76))

        self.btn_play_pause.draw(self.screen, self.font_small)
        self.btn_restart.draw(self.screen, self.font_small)

    def draw_communities(self):
        centers = self.world.community_centers
        if centers is None:
            return
        for i, (cx, cy) in enumerate(centers.astype(int)):
            rect = pygame.Rect(cx, cy, COMMUNITY_SIZE[0], COMMUNITY_SIZE[1])
            pygame.draw.rect(self.screen, COLORS['community'], rect, 1)

    def draw_agents(self):
        a = self.world.agents
        if a.positions is None:
            return
        pos = a.positions.astype(np.int32).tolist()
        for i in range(a.n):
            pygame.draw.circle(self.screen, _STATUS_COLORS[a.statuses[i]],
                               pos[i], 3)

    def draw_quarantine_zone(self):
        qx, qy, qw, qh = QUARANTINE_RECT
        self._dashed_rect((qx, qy, qw, qh), COLORS['quarantine'])
        lbl = self.font_small.render('QUARANTINE', True, COLORS['quarantine'])
        self.screen.blit(lbl, (qx + 5, qy + 5))

    def draw_hud(self):
        with self.state['lock']:
            t = self.state.get('t_days', 0.0)
            u = self.state.get('u_current', 0.0)
        a = self.world.agents
        lines = [
            (f'Day {t:5.1f}', COLORS['text']),
            (f'u*  {u:5.3f}', COLORS['text']),
        ]
        if a.statuses is not None:
            nS = int((a.statuses == Agents.S).sum())
            nI = int((a.statuses == Agents.I).sum())
            nR = int((a.statuses == Agents.R).sum())
            nQ = int((a.statuses == Agents.Q).sum())
            nD = int((a.statuses == Agents.D).sum())
            lines.append((f'S:{nS:3d} I:{nI:3d} R:{nR:3d} Q:{nQ:2d} D:{nD:2d}',
                          COLORS['dim']))
        if self.game_state == GameState.PAUSED:
            lines.append(('[ PAUSED ]', (255, 255, 100)))
        elif self.game_state == GameState.SETUP:
            lines.append(('[ SETUP - adjust params, then PLAY ]',
                          (100, 180, 100)))
        for k, (text, color) in enumerate(lines):
            surf = self.font.render(text, True, color)
            self.screen.blit(surf, (8, 8 + k * 16))

    def draw_sliders(self):
        for s in self._all_sliders:
            s.draw(self.screen, self.font_small)

    # ── Events ──────────────────────────────────────────────────────────────

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    self._do_restart()
                elif event.key == pygame.K_SPACE:
                    self._toggle_play_pause()

            if self.btn_play_pause.handle_event(event):
                self._toggle_play_pause()
            if self.btn_restart.handle_event(event):
                self._do_restart()

            for s in self._all_sliders:
                if s.handle_event(event):
                    if s is not self.slider_speed:
                        self._on_param_changed(s)

    def _toggle_play_pause(self):
        if self.game_state == GameState.SETUP:
            self._set_state(GameState.PLAYING)
        elif self.game_state == GameState.PLAYING:
            self._set_state(GameState.PAUSED)
        elif self.game_state == GameState.PAUSED:
            self._set_state(GameState.PLAYING)

    def _on_param_changed(self, s):
        if s is self.slider_inf_radius:
            _agent.INFECTION_RADIUS = int(s.value)
            with self.state["lock"]:
                self.state["infection_radius"] = int(s.value)
        elif s is self.slider_jump_prob:
            _agent.JUMP_PROB = s.value
            with self.state["lock"]:
                self.state["jump_prob"] = s.value
        elif s is self.slider_q_delay:
            _agent.QUARANTINE_DELAY = s.value
            with self.state["lock"]:
                self.state["quarantine_delay"] = s.value
        elif s is self.slider_omega:
            with self.state["lock"]:
                self.state["omega"] = s.value
        elif s is self.slider_care:
            with self.state["lock"]:
                self.state["care"] = s.value
        elif s is self.slider_urgency:
            with self.state["lock"]:
                self.state["urgency"] = s.value
        elif s is self.slider_mu:
            with self.state["lock"]:
                self.state["mu_fit"] = s.value
        elif s is self.slider_initial_inf:
            self._restart_pending = (
                int(s.value) != self._active_initial_infected)
            with self.state["lock"]:
                self.state["initial_infected_pending"] = int(s.value)
            return

        if s.category == "policy" and self.game_state == GameState.PLAYING:
            self._compute_effective_rates()
            self._schedule_resolve()
        else:
            self._trigger_resolve()

    def _do_restart(self):
        n_inf = int(self.slider_initial_inf.value)
        self.world.agents.initialize(self.world.community_centers,
                                     n_initial_infected=n_inf)
        self._active_initial_infected = n_inf
        self._restart_pending = False
        self.world.t_days = 0.0
        self.world.u_current = 0.0
        i0 = n_inf / self.world.agents.n
        with self.state['lock']:
            self.state['S_history'].clear()
            self.state['I_history'].clear()
            self.state['R_history'].clear()
            self.state.get('D_history', []).clear()
            self.state['t_history'].clear()
            self.state['t_days'] = 0.0
            self.state['u_current'] = 0.0
            self.state['I0'] = i0
            self.state['S0'] = 1.0 - i0
        if self.dashboard:
            self.dashboard.I0 = i0
            self.dashboard.S0 = 1.0 - i0
            self.dashboard.reset_tracking()
        if self.on_restart:
            self.on_restart()
        self._set_state(GameState.SETUP)

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _dashed_rect(self, rect, color, dash=8, gap=5):
        x, y, w, h = rect
        for s, e in [((x, y), (x+w, y)), ((x+w, y), (x+w, y+h)),
                      ((x+w, y+h), (x, y+h)), ((x, y+h), (x, y))]:
            self._dashed_line(s, e, color, dash, gap)

    def _dashed_line(self, start, end, color, dash=8, gap=5):
        dx, dy = end[0]-start[0], end[1]-start[1]
        length = math.sqrt(dx*dx + dy*dy)
        if length == 0:
            return
        ux, uy = dx/length, dy/length
        pos = 0.0
        while pos < length:
            e = min(pos+dash, length)
            pygame.draw.line(self.screen, color,
                             (int(start[0]+ux*pos), int(start[1]+uy*pos)),
                             (int(start[0]+ux*e), int(start[1]+uy*e)), 1)
            pos += dash + gap
