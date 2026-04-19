"""core.py — Graph model, Robot state machine, and Cooperative A* Planner."""

import math
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Any

from utils import PriorityQueue, euclidean, clamp, normalize, reverse_edge_id

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# GRAPH / MAP MODEL
# ══════════════════════════════════════════════════════════════

class Edge:
    """A lane in the factory graph — stateful, capacity-aware."""

    def __init__(self, data: Dict):
        self.id: str = data["id"]
        self.start_node: str = data["from"]
        self.end_node: str = data["to"]
        self.directed: bool = data.get("directed", False)
        self.length: float = data.get("length", 100.0)
        self.max_speed: float = data.get("max_speed", 1.0)
        self.lane_type: str = data.get("lane_type", "normal")
        self.safety_level: int = data.get("safety_level", 1)
        self.capacity: int = data.get("capacity", 2)
        self.is_critical: bool = data.get("is_critical", False)

        # Dynamic state
        self.current_occupancy: int = 0
        self.usage_count: int = 0          # historical heatmap
        self.congestion_score: float = 0.0
        self._occupants: Set[str] = set()
        self._flow_directions: List[str] = []  # "fwd" | "rev"

    # ── occupancy management ──────────────────────────────────

    def can_enter(self, robot_id: str, direction: str = "fwd") -> bool:
        if robot_id in self._occupants:
            return True
        return self.current_occupancy < self.capacity

    def enter(self, robot_id: str, direction: str = "fwd"):
        if robot_id not in self._occupants:
            self._occupants.add(robot_id)
            self.current_occupancy = len(self._occupants)
            self.usage_count += 1
            self._flow_directions.append(direction)

    def exit(self, robot_id: str, direction: str = "fwd"):
        self._occupants.discard(robot_id)
        self.current_occupancy = len(self._occupants)
        if direction in self._flow_directions:
            self._flow_directions.remove(direction)

    def has_contraflow(self, direction: str) -> bool:
        opposite = "rev" if direction == "fwd" else "fwd"
        return opposite in self._flow_directions

    def update_congestion(self, alpha: float = 0.6, beta: float = 0.3, gamma: float = 0.1,
                           max_usage: float = 1000.0):
        occ_ratio = self.current_occupancy / max(self.capacity, 1)
        usage_norm = normalize(self.usage_count, 0, max_usage)
        contraflow_pen = 1.0 if self._flow_directions and len(set(self._flow_directions)) > 1 else 0.0
        self.congestion_score = clamp(
            alpha * occ_ratio + beta * usage_norm + gamma * contraflow_pen, 0.0, 1.0
        )

    def __repr__(self):
        return (f"Edge({self.id} {self.start_node}→{self.end_node} "
                f"occ={self.current_occupancy}/{self.capacity} cong={self.congestion_score:.2f})")


class Node:
    def __init__(self, data: Dict):
        self.id: str = data["id"]
        self.x: float = data.get("x", 0)
        self.y: float = data.get("y", 0)
        self.label: str = data.get("label", self.id)

    def pos(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def __repr__(self):
        return f"Node({self.id} @ {self.x},{self.y})"


class MapGraph:
    """Lane-aware directed graph with full edge state."""

    def __init__(self, map_data: Dict):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}
        # adjacency: node_id → list of (neighbor_id, edge_id)
        self._adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        for nd in map_data["nodes"]:
            self.nodes[nd["id"]] = Node(nd)

        for ed in map_data["edges"]:
            e = Edge(ed)
            self.edges[e.id] = e
            self._adj[e.start_node].append((e.end_node, e.id))
            if not e.directed:
                # create virtual reverse
                rev_id = reverse_edge_id(e.id)
                self.edges[rev_id] = e   # same object, direction tracked via direction param
                self._adj[e.end_node].append((e.start_node, rev_id))

        logger.info(f"MapGraph loaded: {len(self.nodes)} nodes, {len(self.edges)} edges")

    def neighbors(self, node_id: str) -> List[Tuple[str, str]]:
        return self._adj.get(node_id, [])

    def get_edge(self, edge_id: str) -> Optional[Edge]:
        return self.edges.get(edge_id)

    def heuristic(self, n1: str, n2: str) -> float:
        p1 = self.nodes[n1].pos()
        p2 = self.nodes[n2].pos()
        return euclidean(p1, p2)

    def update_all_congestion(self, max_usage: float = 1000.0):
        seen = set()
        for eid, edge in self.edges.items():
            if edge.id not in seen:
                edge.update_congestion(max_usage=max_usage)
                seen.add(edge.id)

    def max_usage_count(self) -> int:
        seen = set()
        mx = 1
        for edge in self.edges.values():
            if edge.id not in seen:
                mx = max(mx, edge.usage_count)
                seen.add(edge.id)
        return mx


# ══════════════════════════════════════════════════════════════
# ROBOT STATE MACHINE
# ══════════════════════════════════════════════════════════════

ROBOT_STATES = {"IDLE", "MOVING", "WAITING", "REPLANNING", "EMERGENCY_STOP", "BLOCKED", "GOAL_REACHED"}


class Robot:
    """Encapsulates robot state, motion, and local safety enforcement."""

    def __init__(self, robot_id: str, start_node: str, goal_node: str,
                 allowed_safety_max: int = 3, config: Dict = None):
        self.id = robot_id
        self.start_node = start_node
        self.goal_node = goal_node
        self.allowed_safety_max = allowed_safety_max
        self.config = config or {}

        # State
        self.state: str = "IDLE"
        self.current_node: str = start_node
        self.next_node: Optional[str] = None
        self.current_edge_id: Optional[str] = None
        self.edge_direction: str = "fwd"   # "fwd" or "rev"

        # Path = list of (node_id, edge_id) tuples
        self.path: List[Tuple[str, str]] = []  # [(next_node, via_edge_id), ...]
        self.path_index: int = 0

        # Motion
        self.speed: float = 0.0
        self.progress: float = 0.0   # 0..1 along current edge
        self.x: float = 0.0
        self.y: float = 0.0

        # Metrics
        self.wait_ticks: int = 0
        self.total_ticks: int = 0
        self.goals_completed: int = 0
        self.stuck_ticks: int = 0
        self.replan_count: int = 0

        # Priority
        self.priority_score: float = 0.0
        self.urgency: float = 1.0

        # Reservation ref (set by traffic module)
        self.reserved_edges: List[str] = []

    def set_path(self, path: List[Tuple[str, str]]):
        """path: list of (next_node, edge_id)"""
        self.path = path
        self.path_index = 0
        if path:
            self.state = "MOVING"
        else:
            self.state = "IDLE"

    def peek_next_edge(self) -> Optional[str]:
        if self.path_index < len(self.path):
            return self.path[self.path_index][1]
        return None

    def peek_next_node(self) -> Optional[str]:
        if self.path_index < len(self.path):
            return self.path[self.path_index][0]
        return None

    def advance_path(self):
        """Move to next step in path."""
        if self.path_index < len(self.path):
            self.current_node = self.path[self.path_index][0]
            self.path_index += 1
            self.current_edge_id = None
            self.progress = 0.0
            self.stuck_ticks = 0
            if self.path_index >= len(self.path):
                if self.current_node == self.goal_node:
                    self.state = "GOAL_REACHED"
                    self.goals_completed += 1

    def progress_ratio(self) -> float:
        """How far along total path (0..1)."""
        if not self.path:
            return 1.0
        return self.path_index / len(self.path)

    def enforce_safe_distance(self, graph: MapGraph, all_robots: List["Robot"], safe_dist: float = 25.0) -> bool:
        """Returns True if it's safe to advance, False if must wait."""
        next_node = self.peek_next_node()
        if next_node is None:
            return True
            
        for other in all_robots:
            if other.id == self.id:
                continue
                
            # Node lock: if someone is stopped exactly at our target node
            if other.current_node == next_node and other.state != "MOVING":
                return False
                
            # Continuous Euclidean distance check
            dist = math.hypot(self.x - other.x, self.y - other.y)
            if dist < safe_dist:
                # If on the same edge and they are ahead of us, we must wait
                if self.current_edge_id and other.current_edge_id == self.current_edge_id:
                    if self.edge_direction == other.edge_direction:
                        if self.progress < other.progress:
                            return False
                    else:
                        return False # Head on
                        
        return True

    def emergency_stop(self, reason: str = ""):
        self.state = "EMERGENCY_STOP"
        self.speed = 0.0
        logger.warning(f"Robot {self.id} EMERGENCY STOP: {reason}")

    def update_position(self, graph: MapGraph, delta: float):
        """Interpolate robot's world position along current edge."""
        if self.current_edge_id:
            edge = graph.get_edge(self.current_edge_id)
            if edge:
                base = graph.get_edge(edge.id)
                if base:
                    if not self.current_edge_id.endswith("_rev"):
                        sx, sy = graph.nodes[edge.start_node].x, graph.nodes[edge.start_node].y
                        ex, ey = graph.nodes[edge.end_node].x, graph.nodes[edge.end_node].y
                    else:
                        sx, sy = graph.nodes[edge.end_node].x, graph.nodes[edge.end_node].y
                        ex, ey = graph.nodes[edge.start_node].x, graph.nodes[edge.start_node].y
                    self.x = sx + (ex - sx) * self.progress
                    self.y = sy + (ey - sy) * self.progress
        else:
            node = graph.nodes.get(self.current_node)
            if node:
                self.x, self.y = node.x, node.y

    def __repr__(self):
        return f"Robot({self.id} @{self.current_node} → {self.goal_node} [{self.state}])"


# ══════════════════════════════════════════════════════════════
# COOPERATIVE A* PLANNER
# ══════════════════════════════════════════════════════════════

class Planner:
    """
    Stateless cooperative A* planner.
    Respects reservation table, safety constraints, and dynamic edge costs.
    """

    def __init__(self, graph: MapGraph, config: Dict = None):
        self.graph = graph
        self.cfg = config or {}
        self.reservation_horizon: int = self.cfg.get("reservation_horizon", 20)

    def plan(self, robot: Robot, reservation_table: Dict, cost_fn,
             current_time: int = 0) -> List[Tuple[str, str]]:
        """
        Returns path as [(next_node, edge_id), ...] from robot.current_node to robot.goal_node.
        Uses time-aware A* with reservation checking.
        """
        start = robot.current_node
        goal = robot.goal_node

        if start == goal:
            return []

        # (f_score, g_score, time_step, node, path_so_far)
        open_set = PriorityQueue()
        h0 = self.graph.heuristic(start, goal)
        open_set.push((start, current_time), h0)

        came_from: Dict = {}   # (node, t) → (prev_node, t-1, edge_id)
        g_score: Dict = {}
        g_score[(start, current_time)] = 0.0

        visited = set()
        max_iter = 2000

        while not open_set.empty() and max_iter > 0:
            max_iter -= 1
            (node, t), f = open_set.pop()

            if node == goal:
                return self._reconstruct_path(came_from, (node, t))

            state = (node, t)
            if state in visited:
                continue
            visited.add(state)

            for neighbor, edge_id in self.graph.neighbors(node):
                edge = self.graph.get_edge(edge_id)
                if edge is None:
                    continue

                # Safety constraint
                if edge.safety_level > robot.allowed_safety_max:
                    continue

                direction = "rev" if edge_id.endswith("_rev") else "fwd"
                next_t = t + 1

                # Reservation check
                base_eid = edge_id.replace("_rev", "")
                if self._is_reserved(base_eid, next_t, robot.id, reservation_table):
                    continue

                # Capacity check
                if not edge.can_enter(robot.id, direction):
                    continue

                edge_cost = cost_fn(edge, t, robot)
                if edge_cost == float("inf"):
                    continue

                tent_g = g_score.get(state, float("inf")) + edge_cost

                next_state = (neighbor, next_t)
                if tent_g < g_score.get(next_state, float("inf")):
                    g_score[next_state] = tent_g
                    came_from[next_state] = (node, t, edge_id)
                    f_new = tent_g + self.graph.heuristic(neighbor, goal)
                    open_set.push(next_state, f_new)

        logger.warning(f"Planner: No path found for {robot.id} from {start} to {goal}")
        return []

    def _is_reserved(self, edge_id: str, time_step: int, robot_id: str,
                      reservation_table: Dict) -> bool:
        entry = reservation_table.get((edge_id, time_step))
        return entry is not None and entry != robot_id

    def _reconstruct_path(self, came_from: Dict, end_state: Tuple) -> List[Tuple[str, str]]:
        path = []
        state = end_state
        while state in came_from:
            prev_node, prev_t, edge_id = came_from[state]
            node = state[0]
            path.append((node, edge_id))
            state = (prev_node, prev_t)
        path.reverse()
        return path

    def replan(self, robot: Robot, reason: str, reservation_table: Dict,
               cost_fn, current_time: int) -> List[Tuple[str, str]]:
        robot.replan_count += 1
        robot.state = "REPLANNING"
        logger.info(f"Replanning {robot.id} | reason={reason} | t={current_time}")
        new_path = self.plan(robot, reservation_table, cost_fn, current_time)
        return new_path
