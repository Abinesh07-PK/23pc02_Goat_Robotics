"""visualizer.py — Pygame-based real-time simulation renderer."""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logger.warning("pygame not installed — visualization disabled. Run: pip install pygame")

from utils import (
    LANE_TYPE_COLORS, ROBOT_COLORS, STATE_COLORS,
    congestion_color, heatmap_color, clamp, normalize
)


# ══════════════════════════════════════════════════════════════
# VISUALIZER
# ══════════════════════════════════════════════════════════════

class Visualizer:
    """
    Pygame renderer for the multi-robot traffic simulation.

    Renders:
      - Graph edges (colored by lane type / congestion / heatmap)
      - Nodes (waypoints)
      - Robot positions with state color coding
      - Robot paths (planned trajectory)
      - Reserved lanes highlight
      - HUD: tick, robot states, metrics
      - Legend
    """

    # Display mode constants
    MODE_NORMAL    = 0
    MODE_HEATMAP   = 1
    MODE_CONGESTION = 2

    def __init__(self, config: Dict):
        if not PYGAME_AVAILABLE:
            self._active = False
            return
        self._active = True

        vis_cfg = config.get("visualization", {})
        self.width:        int   = vis_cfg.get("window_width",  1200)
        self.height:       int   = vis_cfg.get("window_height",  800)
        self.fps:          int   = vis_cfg.get("fps", 30)
        self.node_r:       int   = vis_cfg.get("node_radius",    12)
        self.robot_r:      int   = vis_cfg.get("robot_radius",    8)
        self.show_heatmap: bool  = vis_cfg.get("show_heatmap",  True)
        self.show_res:     bool  = vis_cfg.get("show_reservations", True)
        self.show_paths:   bool  = vis_cfg.get("show_paths",    True)
        self.show_ids:     bool  = vis_cfg.get("show_robot_ids", True)

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Multi-Robot Traffic Control — GOAT Hackathon")
        self.clock  = pygame.time.Clock()

        # Fonts
        self.font_sm  = pygame.font.SysFont("monospace", 11)
        self.font_md  = pygame.font.SysFont("monospace", 13)
        self.font_lg  = pygame.font.SysFont("monospace", 16, bold=True)

        # Colors
        self.BG         = (18, 18, 24)
        self.GRID       = (30, 30, 40)
        self.NODE_CLR   = (70, 90, 120)
        self.NODE_LABEL = (140, 160, 200)
        self.TEXT_CLR   = (200, 210, 230)
        self.HUD_BG     = (20, 20, 30, 200)
        self.WHITE      = (255, 255, 255)
        self.RED        = (220, 60, 60)
        self.YELLOW     = (240, 200, 50)
        self.GREEN      = (60, 200, 100)

        # Display mode toggle
        self.display_mode = self.MODE_NORMAL

        # Graph drawing bounds (with margin)
        self._margin = 80
        self._scale  = 1.0
        self._offset = (0, 0)

        logger.info("Visualizer initialised.")

    def is_active(self) -> bool:
        return self._active

    # ── coordinate mapping ────────────────────────────────────

    def _map_pos(self, x: float, y: float, graph) -> Tuple[int, int]:
        """Map graph coordinates to screen coordinates."""
        xs = [n.x for n in graph.nodes.values()]
        ys = [n.y for n in graph.nodes.values()]
        if not xs:
            return (int(x), int(y))
        mx, my = min(xs), min(ys)
        Rx = max(xs) - mx or 1
        Ry = max(ys) - my or 1

        # Preserve aspect ratio
        avail_w = self.width  - self._margin * 2 - 300  # reserve right panel
        avail_h = self.height - self._margin * 2
        scale = min(avail_w / Rx, avail_h / Ry)

        sx = int((x - mx) * scale + self._margin)
        sy = int((y - my) * scale + self._margin)
        return (sx, sy)

    # ── main render ───────────────────────────────────────────

    def render(self, graph, robots: List, tick: int, metrics=None):
        if not self._active:
            return

        self.screen.fill(self.BG)
        self._draw_grid()

        max_usage = graph.max_usage_count()

        # Edges
        self._draw_edges(graph, max_usage)

        # Nodes
        self._draw_nodes(graph)

        # Robot paths
        if self.show_paths:
            self._draw_paths(robots, graph)

        # Robots
        self._draw_robots(robots, graph)

        # HUD
        self._draw_hud(robots, tick, metrics)

        # Legend
        self._draw_legend(graph)

        pygame.display.flip()
        self.clock.tick(self.fps)

    # ── drawing helpers ───────────────────────────────────────

    def _draw_grid(self):
        for x in range(0, self.width, 60):
            pygame.draw.line(self.screen, self.GRID, (x, 0), (x, self.height), 1)
        for y in range(0, self.height, 60):
            pygame.draw.line(self.screen, self.GRID, (0, y), (self.width, y), 1)

    def _draw_edges(self, graph, max_usage: float):
        drawn = set()
        reservation_table = {}  # not passed — skip reservation highlight here

        for eid, edge in graph.edges.items():
            if edge.id in drawn:
                continue
            drawn.add(edge.id)

            sn = graph.nodes.get(edge.start_node)
            en = graph.nodes.get(edge.end_node)
            if not sn or not en:
                continue

            sp = self._map_pos(sn.x, sn.y, graph)
            ep = self._map_pos(en.x, en.y, graph)

            # Choose color based on display mode
            if self.display_mode == self.MODE_HEATMAP:
                color = heatmap_color(edge.usage_count, max_usage)
            elif self.display_mode == self.MODE_CONGESTION:
                color = congestion_color(edge.congestion_score)
            else:
                color = LANE_TYPE_COLORS.get(edge.lane_type, (100, 100, 120))

            # Line width by congestion
            width = 3 + int(edge.congestion_score * 4)

            # Critical lane → dashed effect via segmented line
            if edge.is_critical:
                self._draw_dashed_line(sp, ep, color, width + 1)
            else:
                pygame.draw.line(self.screen, color, sp, ep, width)

            # Occupancy indicator on edge midpoint
            mid = ((sp[0] + ep[0]) // 2, (sp[1] + ep[1]) // 2)
            if edge.current_occupancy > 0:
                occ_text = self.font_sm.render(str(edge.current_occupancy), True, self.YELLOW)
                self.screen.blit(occ_text, (mid[0] - 4, mid[1] - 6))

    def _draw_dashed_line(self, start: Tuple, end: Tuple, color, width: int, dash: int = 12):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = max((dx**2 + dy**2) ** 0.5, 1)
        steps = int(length / dash)
        for i in range(steps):
            t0 = i / steps
            t1 = (i + 0.5) / steps
            p0 = (int(start[0] + dx * t0), int(start[1] + dy * t0))
            p1 = (int(start[0] + dx * t1), int(start[1] + dy * t1))
            pygame.draw.line(self.screen, color, p0, p1, width)

    def _draw_nodes(self, graph):
        for nid, node in graph.nodes.items():
            pos = self._map_pos(node.x, node.y, graph)
            pygame.draw.circle(self.screen, self.NODE_CLR, pos, self.node_r)
            pygame.draw.circle(self.screen, (100, 130, 180), pos, self.node_r, 2)
            label = self.font_sm.render(nid, True, self.NODE_LABEL)
            self.screen.blit(label, (pos[0] - label.get_width() // 2, pos[1] + self.node_r + 2))

    def _draw_paths(self, robots: List, graph):
        for i, robot in enumerate(robots):
            if not robot.path or robot.state == "GOAL_REACHED":
                continue
            color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            dim_color = tuple(max(c - 80, 0) for c in color)
            prev_pos = self._map_pos(
                graph.nodes[robot.current_node].x,
                graph.nodes[robot.current_node].y, graph
            )
            for step_idx in range(robot.path_index, min(robot.path_index + 8, len(robot.path))):
                next_node_id, _ = robot.path[step_idx]
                nnode = graph.nodes.get(next_node_id)
                if nnode:
                    npos = self._map_pos(nnode.x, nnode.y, graph)
                    pygame.draw.line(self.screen, dim_color, prev_pos, npos, 1)
                    prev_pos = npos

    def _draw_robots(self, robots: List, graph):
        for i, robot in enumerate(robots):
            color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            state_color = STATE_COLORS.get(robot.state, color)

            rx, ry = int(robot.x), int(robot.y)
            sx, sy = self._map_pos(rx, ry, graph)

            # Outer ring (state color)
            pygame.draw.circle(self.screen, state_color, (sx, sy), self.robot_r + 3)
            # Inner fill (robot color)
            pygame.draw.circle(self.screen, color, (sx, sy), self.robot_r)

            # Robot ID label
            if self.show_ids:
                label = self.font_sm.render(robot.id, True, self.WHITE)
                self.screen.blit(label, (sx - label.get_width() // 2, sy - self.robot_r - 14))

            # Goal marker (small diamond)
            goal_node = graph.nodes.get(robot.goal_node)
            if goal_node:
                gx, gy = self._map_pos(goal_node.x, goal_node.y, graph)
                pts = [(gx, gy - 8), (gx + 6, gy), (gx, gy + 8), (gx - 6, gy)]
                pygame.draw.polygon(self.screen, color, pts, 2)

    def _draw_hud(self, robots: List, tick: int, metrics=None):
        """Right-side HUD panel."""
        panel_x = self.width - 290
        panel_rect = pygame.Rect(panel_x - 10, 0, 300, self.height)

        # Semi-transparent background
        hud_surf = pygame.Surface((300, self.height), pygame.SRCALPHA)
        hud_surf.fill((15, 15, 25, 220))
        self.screen.blit(hud_surf, (panel_x - 10, 0))

        y = 15
        def text(msg, color=None, font=None, bold=False):
            nonlocal y
            c = color or self.TEXT_CLR
            f = font or self.font_md
            surf = f.render(msg, True, c)
            self.screen.blit(surf, (panel_x, y))
            y += surf.get_height() + 3

        text(f"TICK: {tick}", self.GREEN, self.font_lg)
        text("─" * 26, (60, 60, 80))

        # Display mode indicator
        modes = ["NORMAL", "HEATMAP", "CONGESTION"]
        text(f"View: {modes[self.display_mode]}", self.YELLOW)
        text("  [H] Heatmap  [C] Congestion", (120, 120, 150), self.font_sm)
        text("  [Space] Pause  [Q] Quit", (120, 120, 150), self.font_sm)
        text("─" * 26, (60, 60, 80))

        text("ROBOTS:", self.TEXT_CLR, self.font_lg)
        for i, robot in enumerate(robots):
            color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            sc = STATE_COLORS.get(robot.state, color)
            state_str = robot.state[:10].ljust(10)
            line = f"{robot.id}: {state_str} G:{robot.goals_completed}"
            text(line, sc, self.font_sm)

        text("─" * 26, (60, 60, 80))

        if metrics:
            s = metrics.summary()
            text("METRICS:", self.TEXT_CLR, self.font_lg)
            text(f"Goals: {s['total_goals_completed']}", self.GREEN)
            text(f"Replans: {s['total_replans']}", self.YELLOW)
            text(f"Deadlocks: {s['deadlocks_detected']}", self.RED)
            text(f"Throughput: {s['throughput_per_100_ticks']:.2f}/100t", self.TEXT_CLR)

    def _draw_legend(self, graph):
        """Bottom legend for lane types."""
        x, y = 15, self.height - 90
        self.screen.blit(self.font_md.render("LANE TYPES:", True, self.TEXT_CLR), (x, y))
        y += 18
        for lane_type, color in LANE_TYPE_COLORS.items():
            pygame.draw.rect(self.screen, color, (x, y, 16, 10))
            self.screen.blit(self.font_sm.render(lane_type, True, self.TEXT_CLR), (x + 22, y - 1))
            x += 130

    # ── event handling ────────────────────────────────────────

    def handle_events(self, simulator) -> bool:
        """
        Process pygame events.
        Returns False if simulation should stop.
        """
        if not self._active:
            return True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return False
                if event.key == pygame.K_SPACE:
                    simulator.pause()
                if event.key == pygame.K_h:
                    self.display_mode = self.MODE_HEATMAP if self.display_mode != self.MODE_HEATMAP else self.MODE_NORMAL
                if event.key == pygame.K_c:
                    self.display_mode = self.MODE_CONGESTION if self.display_mode != self.MODE_CONGESTION else self.MODE_NORMAL
                if event.key == pygame.K_r:
                    self.display_mode = self.MODE_NORMAL
        return True

    def quit(self):
        if self._active and PYGAME_AVAILABLE:
            pygame.quit()
