"""intelligence.py — Cost engine, policy engine, priority engine, flow allocator."""

import math
import logging
from typing import Dict, List, Optional, Tuple, Any

from utils import clamp, normalize

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# COST ENGINE
# ══════════════════════════════════════════════════════════════

class CostEngine:
    """
    Composite, time-aware edge cost function.

    cost(e,t) = wd*d(e) + α*c_cong + β*c_safety + γ*c_contraflow + δ*c_heatmap
    """

    def __init__(self, config: Dict):
        w = config.get("weights", {})
        self.wd    = w.get("distance",    1.0)
        self.alpha = w.get("congestion",  1.2)
        self.beta  = w.get("safety",      2.0)
        self.gamma = w.get("contraflow",  3.0)
        self.delta = w.get("heatmap",     0.8)
        self.k     = config.get("safety_exp_k", 1.5)
        self._max_usage: float = 1.0    # updated each tick

    def update_max_usage(self, max_usage: float):
        self._max_usage = max(max_usage, 1.0)

    def compute(self, edge, time_step: int, robot) -> float:
        """
        Returns edge traversal cost. Returns INF for forbidden edges.
        `edge` is a core.Edge instance; `robot` is a core.Robot instance.
        """
        from core import Edge  # avoid circular at top-level

        # Hard constraint: forbidden edges → INF
        if edge.safety_level > robot.allowed_safety_max:
            return float("inf")
        if edge.current_occupancy >= edge.capacity:
            # Only hard-block if robot is not already in this edge
            if robot.id not in getattr(edge, "_occupants", set()):
                return float("inf")

        # 1. Distance cost: traversal time
        d_cost = edge.length / max(edge.max_speed, 0.1)

        # 2. Congestion cost (real-time) [0,1]
        c_cong = edge.congestion_score  # already normalised in Edge.update_congestion

        # 3. Safety cost (nonlinear penalty)
        # Normalise safety_level to [0,1] assuming max level = 5
        s_norm = edge.safety_level / 5.0
        c_safety = math.exp(self.k * s_norm) - 1.0  # starts near 0 for level 1

        # 4. Contraflow cost
        direction = "rev" if getattr(edge, "_current_direction", "fwd") == "rev" else "fwd"
        c_contraflow = 1.0 if edge.has_contraflow(direction) else 0.0

        # 5. Heatmap cost (historical load)
        c_heatmap = normalize(edge.usage_count, 0, self._max_usage)

        cost = (
            self.wd    * d_cost +
            self.alpha * c_cong +
            self.beta  * c_safety +
            self.gamma * c_contraflow +
            self.delta * c_heatmap
        )
        return max(cost, 0.01)

    def make_cost_fn(self):
        """Returns a callable (edge, time_step, robot) → float for use by Planner."""
        def fn(edge, time_step, robot):
            return self.compute(edge, time_step, robot)
        return fn


# ══════════════════════════════════════════════════════════════
# POLICY ENGINE
# ══════════════════════════════════════════════════════════════

class PolicyEngine:
    """
    Controls WHEN robots replan and HOW they react (wait vs reroute).
    """

    def __init__(self, config: Dict):
        self.replan_interval_high: int = config.get("replan_interval_high", 50)
        self.replan_interval_low:  int = config.get("replan_interval_low",  10)
        self.stuck_threshold:      int = config.get("stuck_threshold",       30)
        self.wait_delay_threshold: int = config.get("wait_delay_threshold",  10)

    def should_replan(self, robot, system_state: Dict, current_tick: int) -> Tuple[bool, str]:
        """
        Returns (should_replan, reason).
        Checks event-driven conditions first, then periodic.
        """
        # Event-driven triggers (immediate)
        if system_state.get("reservation_denied", {}).get(robot.id, False):
            return True, "reservation_denied"

        if system_state.get("deadlock_participants", set()) and robot.id in system_state.get("deadlock_participants", set()):
            return True, "deadlock_participant"

        if robot.stuck_ticks >= self.stuck_threshold:
            return True, "stuck"

        if system_state.get("congestion_hotspots"):
            # If robot's next edge is a hotspot, replan
            next_eid = robot.peek_next_edge()
            if next_eid and next_eid.replace("_rev", "") in system_state.get("congestion_hotspots", set()):
                return True, "congestion_hotspot"

        # Periodic high-level replan
        if current_tick % self.replan_interval_high == 0 and robot.state not in ("GOAL_REACHED", "EMERGENCY_STOP"):
            return True, "periodic_high"

        return False, ""

    def decide_wait_or_reroute(self, robot, system_state: Dict) -> str:
        """
        Returns "WAIT" or "REROUTE".
        """
        # Check if safe to wait
        is_safe = self._safe_to_wait(robot, system_state)
        delay_ok = robot.wait_ticks < self.wait_delay_threshold
        in_deadlock = robot.id in system_state.get("deadlock_participants", set())
        reservation_denied = system_state.get("reservation_denied", {}).get(robot.id, False)

        if in_deadlock or reservation_denied:
            return "REROUTE"
        if is_safe and delay_ok:
            return "WAIT"
        return "REROUTE"

    def _safe_to_wait(self, robot, system_state: Dict) -> bool:
        # Not in a deadlock, has safe following distance
        if robot.id in system_state.get("deadlock_participants", set()):
            return False
        # Check next edge safety level (if critical/human zone, don't block)
        next_edge_safety = system_state.get("next_edge_safety", {}).get(robot.id, 1)
        return next_edge_safety <= 3

    def lane_forbidden(self, edge, robot) -> bool:
        return edge.safety_level > robot.allowed_safety_max

    def lane_heavily_penalised(self, edge, threshold: float = 0.75) -> bool:
        return edge.congestion_score > threshold


# ══════════════════════════════════════════════════════════════
# PRIORITY ENGINE
# ══════════════════════════════════════════════════════════════

class PriorityEngine:
    """
    Heuristic-based multi-robot priority scoring.

    priority_score = w1 * wait_time + w2 * progress_to_goal + w3 * urgency
    """

    def __init__(self, config: Dict):
        self.w1 = config.get("w1_wait_time",  0.4)
        self.w2 = config.get("w2_progress",   0.4)
        self.w3 = config.get("w3_urgency",    0.2)

    def compute_score(self, robot, max_wait: float = 1.0) -> float:
        wait_norm     = normalize(robot.wait_ticks, 0, max(max_wait, 1))
        progress      = robot.progress_ratio()
        urgency_norm  = clamp(robot.urgency / 10.0, 0.0, 1.0)
        score = self.w1 * wait_norm + self.w2 * progress + self.w3 * urgency_norm
        robot.priority_score = score
        return score

    def rank(self, robots: List) -> List:
        """Return robots sorted by descending priority score."""
        max_wait = max((r.wait_ticks for r in robots), default=1)
        for robot in robots:
            self.compute_score(robot, max_wait=max_wait)
        return sorted(robots, key=lambda r: r.priority_score, reverse=True)

    def lowest_priority(self, robots: List):
        """Returns the lowest-priority robot (deadlock victim candidate)."""
        if not robots:
            return None
        ranked = self.rank(robots)
        return ranked[-1] if ranked else None

    def highest_priority(self, robots: List):
        if not robots:
            return None
        return self.rank(robots)[0]


# ══════════════════════════════════════════════════════════════
# FLOW ALLOCATOR (lightweight cluster routing)
# ══════════════════════════════════════════════════════════════

class FlowAllocator:
    """
    Optional macro-level routing to reduce systemic congestion.
    Runs every 3-5 seconds (not every tick).
    """

    def __init__(self, graph, update_interval: int = 40):
        self.graph = graph
        self.update_interval = update_interval
        self._last_update: int = -9999
        self._assignments: Dict[str, List[str]] = {}  # robot_id → preferred_node_ids

    def update(self, robots: List, current_tick: int):
        if current_tick - self._last_update < self.update_interval:
            return
        self._last_update = current_tick
        self._reassign(robots)

    def _reassign(self, robots: List):
        """
        Simple strategy: identify high-congestion corridors and steer
        robots away from them via preferred node sets.
        """
        # Find most congested edges
        congested: set = set()
        seen = set()
        for eid, edge in self.graph.edges.items():
            if edge.id in seen:
                continue
            seen.add(edge.id)
            if edge.congestion_score > 0.7:
                congested.add(edge.start_node)
                congested.add(edge.end_node)

        for robot in robots:
            # Preferred nodes = all nodes minus highly congested ones
            preferred = [n for n in self.graph.nodes if n not in congested]
            self._assignments[robot.id] = preferred

    def get_preferred_nodes(self, robot_id: str) -> Optional[List[str]]:
        return self._assignments.get(robot_id)

    def get_node_bias(self, robot_id: str, node_id: str) -> float:
        """Returns a small additive cost bias if node is NOT preferred."""
        preferred = self._assignments.get(robot_id)
        if preferred is None:
            return 0.0
        return 0.0 if node_id in preferred else 0.5
