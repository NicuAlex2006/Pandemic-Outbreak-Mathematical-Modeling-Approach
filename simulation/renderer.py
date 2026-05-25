import math
import pygame
import numpy as np

import simulation.agent as _agent
from simulation.agent import Agents, COMMUNITY_SIZE, QUARANTINE_RECT

SIM_W = 900
DASHBOARD_W = 550
WINDOW_W = SIM_W + DASHBOARD_W
WINDOW_H = 700
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
}

# Indexed by status code: S=0, I=1, R=2, Q=3, D=4
_STATUS_COLORS = [COLORS['S'], COLORS['I'], COLORS['R'], COLORS['Q'], COLORS['D']]

BASE_INFECTION_RADIUS = _agent.INFECTION_RADIUS
BASE_JUMP_PROB = _agent.JUMP_PROB
BASE_QUARANTINE_DELAY = _agent.QUARANTINE_DELAY


class Slider:
    def __init__(self, x, y, w, label, min_val, max_val, initial, fmt=".2f"):
        self.x, self.y, self.w = x, y, w
        self.label = label
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.value = float(initial)
        self.fmt = fmt
        self.dragging = False
        self.handle_r = 7
        self.track = None

    def setup(self):
        self.track = pygame.Rect(self.x, self.y + 12, self.w, 3)

    def _handle_x(self):
        frac = (self.value - self.min_val) / (self.max_val - self.min_val)
        return int(self.x + frac * self.w)

    def draw(self, surface, font):
        pygame.draw.rect(surface, (75, 75, 100), self.track)
        hx = self._handle_x()
        pygame.draw.circle(surface, (160, 160, 210), (hx, self.y + 13), self.handle_r)
        txt = font.render(f'{self.label}: {self.value:{self.fmt}}', True, COLORS['dim'])
        surface.blit(txt, (self.x, self.y - 2))

    def handle_event(self, event) -> bool:
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
                 color=(70, 35, 35), hover_color=(120, 55, 55), border_color=(180, 70, 70)):
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
        self.running = False
        self.on_step = None
        self.on_restart = None

        fitted_mu = dashboard.mu if dashboard else 0.003

        self.btn_restart = Button(760, 8, 110, 24, 'RESTART')
        self.btn_pause = Button(760, 38, 110, 24, 'PAUSE',
                                color=(35, 55, 70), hover_color=(55, 85, 120),
                                border_color=(70, 130, 180))

        sw = 110
        # Row 1
        self.slider_alpha = Slider(20, 620, 170, 'alpha', 0.01, 5.0, 1.0)
        self.slider_speed = Slider(210, 620, sw, 'speed', 1.0, 8.0, 2.0)
        self.slider_mu = Slider(340, 620, sw, 'mu', 0.0, 0.05,
                                 fitted_mu, fmt=".4f")
        # Row 2
        self.slider_inf_radius = Slider(20, 650, sw, 'inf_rad', 1, 15,
                                        BASE_INFECTION_RADIUS, fmt=".0f")
        self.slider_jump_prob = Slider(150, 650, sw, 'jump_p', 0.0, 0.05,
                                       BASE_JUMP_PROB, fmt=".4f")
        self.slider_q_delay = Slider(280, 650, sw, 'q_delay', 0.5, 10.0,
                                      BASE_QUARANTINE_DELAY, fmt=".1f")
        self.slider_omega = Slider(410, 650, sw, 'omega', 0.0, 0.1,
                                    0.0, fmt=".4f")

        self._all_sliders = [
            self.slider_alpha, self.slider_speed, self.slider_mu,
            self.slider_inf_radius, self.slider_jump_prob,
            self.slider_q_delay, self.slider_omega,
        ]

    def setup(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption('Epidemic Control — SIRD-S')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('consolas', 13)
        self.font_small = pygame.font.SysFont('consolas', 11)
        for s in self._all_sliders:
            s.setup()

    def _compute_effective_rates(self):
        if not self.dashboard:
            return
        r_ratio = self.slider_inf_radius.value / max(BASE_INFECTION_RADIUS, 1)
        j_ratio = self.slider_jump_prob.value / max(BASE_JUMP_PROB, 1e-6)
        q_ratio = BASE_QUARANTINE_DELAY / max(self.slider_q_delay.value, 0.1)

        self.dashboard.beta_eff = self.dashboard.beta * r_ratio * (0.5 + 0.5 * j_ratio)
        self.dashboard.gamma_eff = self.dashboard.gamma * q_ratio
        self.dashboard.omega = self.slider_omega.value
        self.dashboard.mu = self.slider_mu.value

        with self.state["lock"]:
            self.state["omega"] = self.slider_omega.value
            self.state["mu_fit"] = self.slider_mu.value

    def _trigger_resolve(self):
        if not self.dashboard:
            return
        self._compute_effective_rates()
        self.dashboard.request_solve(
            self.slider_alpha.value,
            self.dashboard.beta_eff,
            self.dashboard.gamma_eff,
            self.dashboard.mu,
        )

    def run(self):
        self.running = True
        self._compute_effective_rates()
        while self.running:
            self.handle_events()
            if not (self.dashboard and self.dashboard.paused):
                steps = max(1, int(self.slider_speed.value))
                for _ in range(steps):
                    u = self._get_u()
                    self.world.set_u(u)
                    self.world.step()
                    if self.on_step:
                        self.on_step()
                    S, I, R = self.world.get_sir()
                    if self.dashboard:
                        self.dashboard.accumulate(self.world.t_days, S, I)
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

    def draw(self):
        self.screen.fill(COLORS['bg'])
        self.draw_quarantine_zone()
        self.draw_communities()
        self.draw_agents()
        self.draw_hud()
        self.draw_sliders()
        self.btn_restart.draw(self.screen, self.font_small)
        self.btn_pause.draw(self.screen, self.font_small)
        if self.dashboard:
            surf = self.dashboard.render_to_surface(self.state)
            if surf:
                self.screen.blit(surf, (SIM_W, 0))

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
            pygame.draw.circle(self.screen, _STATUS_COLORS[a.statuses[i]], pos[i], 3)

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
            lines.append((f'S:{nS:3d} I:{nI:3d} R:{nR:3d} Q:{nQ:2d} D:{nD:2d}', COLORS['dim']))
        if self.dashboard and self.dashboard.paused:
            lines.append(('[ PAUSED ]', (255, 255, 100)))
        for k, (text, color) in enumerate(lines):
            surf = self.font.render(text, True, color)
            self.screen.blit(surf, (8, 8 + k * 16))

    def draw_sliders(self):
        for s in self._all_sliders:
            s.draw(self.screen, self.font_small)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    self._restart()
                elif event.key == pygame.K_SPACE:
                    self._toggle_pause()

            if self.btn_restart.handle_event(event):
                self._restart()
            if self.btn_pause.handle_event(event):
                self._toggle_pause()

            for s in self._all_sliders:
                if s.handle_event(event):
                    if s is not self.slider_speed:
                        self._on_param_changed(s)

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
        elif s is self.slider_alpha:
            with self.state["lock"]:
                self.state["alpha"] = s.value
        elif s is self.slider_mu:
            with self.state["lock"]:
                self.state["mu_fit"] = s.value

        self._trigger_resolve()

    def _toggle_pause(self):
        if self.dashboard:
            self.dashboard.paused = not self.dashboard.paused
            self.btn_pause.label = 'PLAY' if self.dashboard.paused else 'PAUSE'

    def _restart(self):
        self.world.agents.initialize(self.world.community_centers)
        self.world.t_days = 0.0
        self.world.u_current = 0.0
        with self.state['lock']:
            self.state['S_history'].clear()
            self.state['I_history'].clear()
            self.state['R_history'].clear()
            self.state.get('D_history', []).clear()
            self.state['t_history'].clear()
            self.state['t_days'] = 0.0
            self.state['u_current'] = 0.0
        if self.dashboard:
            self.dashboard.reset_tracking()
        if self.on_restart:
            self.on_restart()

    def _dashed_rect(self, rect, color, dash=8, gap=5):
        x, y, w, h = rect
        for s, e in [((x,y),(x+w,y)),((x+w,y),(x+w,y+h)),
                      ((x+w,y+h),(x,y+h)),((x,y+h),(x,y))]:
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
