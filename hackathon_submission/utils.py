"""utils.py — Shared helper functions and data structures."""

import math
import time
import json
import logging
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Math / geometry helpers
# ─────────────────────────────────────────────

def euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def normalize(value: float, min_val: float, max_val: float) -> float:
    if max_val == min_val:
        return 0.0
    return clamp((value - min_val) / (max_val - min_val), 0.0, 1.0)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * clamp(t, 0.0, 1.0)


# ─────────────────────────────────────────────
# Graph / path helpers
# ─────────────────────────────────────────────

def reverse_edge_id(edge_id: str) -> str:
    """Convention: 'E5_rev' marks the reverse of edge E5."""
    if edge_id.endswith("_rev"):
        return edge_id[:-4]
    return edge_id + "_rev"


def path_length(path: List[str], edges: Dict[str, Any]) -> float:
    total = 0.0
    for eid in path:
        base = eid.replace("_rev", "")
        if base in edges:
            total += edges[base].length
    return total


# ─────────────────────────────────────────────
# Priority Queue (min-heap) with update support
# ─────────────────────────────────────────────

import heapq


class PriorityQueue:
    def __init__(self):
        self._heap: List[Tuple] = []
        self._entry_finder: Dict[Any, List] = {}
        self._REMOVED = object()
        self._counter = 0

    def push(self, item: Any, priority: float):
        if item in self._entry_finder:
            self._remove(item)
        entry = [priority, self._counter, item]
        self._counter += 1
        self._entry_finder[item] = entry
        heapq.heappush(self._heap, entry)

    def _remove(self, item: Any):
        entry = self._entry_finder.pop(item)
        entry[-1] = self._REMOVED

    def pop(self) -> Tuple[Any, float]:
        while self._heap:
            priority, _, item = heapq.heappop(self._heap)
            if item is not self._REMOVED:
                del self._entry_finder[item]
                return item, priority
        raise KeyError("pop from empty PriorityQueue")

    def __len__(self):
        return len(self._entry_finder)

    def __contains__(self, item):
        return item in self._entry_finder

    def empty(self) -> bool:
        return len(self._entry_finder) == 0


# ─────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> Dict:
    try:
        import yaml
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except ImportError:
        # Fallback: minimal defaults if PyYAML not installed
        logger.warning("PyYAML not found; using built-in defaults.")
        return _default_config()
    except FileNotFoundError:
        logger.warning(f"Config file '{path}' not found; using defaults.")
        return _default_config()


def _default_config() -> Dict:
    return {
        "simulation": {"num_robots": 10, "max_time_steps": 10000, "delta_time": 0.1, "tick_rate": 10},
        "planner": {"algorithm": "cooperative_astar", "reservation_horizon": 20,
                    "replan_interval_high": 50, "replan_interval_low": 10, "stuck_threshold": 30},
        "traffic": {"deadlock_check_interval": 5, "congestion_hotspot_threshold": 0.75,
                    "critical_lane_reservation": True},
        "intelligence": {
            "weights": {"distance": 1.0, "congestion": 1.2, "safety": 2.0, "contraflow": 3.0, "heatmap": 0.8},
            "safety_exp_k": 1.5, "wait_delay_threshold": 10
        },
        "priority": {"w1_wait_time": 0.4, "w2_progress": 0.4, "w3_urgency": 0.2},
        "visualization": {
            "window_width": 1200, "window_height": 800, "fps": 30,
            "node_radius": 12, "robot_radius": 8,
            "show_heatmap": True, "show_reservations": True,
            "show_paths": True, "show_robot_ids": True
        },
        "logging": {"log_file": "simulation.log", "metrics_file": "metrics.json", "log_level": "INFO"}
    }


def load_map(path: str = "map.json") -> Dict:
    with open(path, "r") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────

def setup_logging(log_file: str = "simulation.log", level: str = "INFO"):
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


# ─────────────────────────────────────────────
# Event system
# ─────────────────────────────────────────────

class EventBus:
    """Simple pub/sub event bus used across modules."""

    def __init__(self):
        self._subscribers: Dict[str, List] = defaultdict(list)
        self._queue: deque = deque()

    def subscribe(self, event_type: str, callback):
        self._subscribers[event_type].append(callback)

    def publish(self, event_type: str, payload: Any = None):
        self._queue.append((event_type, payload))

    def flush(self):
        while self._queue:
            event_type, payload = self._queue.popleft()
            for cb in self._subscribers.get(event_type, []):
                try:
                    cb(payload)
                except Exception as e:
                    logger.error(f"Event handler error [{event_type}]: {e}")


# ─────────────────────────────────────────────
# Color helpers for visualization
# ─────────────────────────────────────────────

LANE_TYPE_COLORS = {
    "normal":       (100, 180, 100),
    "narrow":       (220, 160,  60),
    "intersection": (100, 160, 220),
    "human_zone":   (220,  80,  80),
}

ROBOT_COLORS = [
    (52,  152, 219),  # blue
    (46,  204, 113),  # green
    (231, 76,  60),   # red
    (155, 89,  182),  # purple
    (241, 196, 15),   # yellow
    (26,  188, 156),  # teal
    (230, 126, 34),   # orange
    (189, 195, 199),  # silver
    (52,  73,  94),   # dark
    (243, 156, 18),   # amber
]

STATE_COLORS = {
    "IDLE":           (150, 150, 150),
    "MOVING":         (46,  204, 113),
    "WAITING":        (241, 196, 15),
    "REPLANNING":     (52,  152, 219),
    "EMERGENCY_STOP": (231, 76,  60),
    "BLOCKED":        (192, 57,  43),
    "GOAL_REACHED":   (39,  174, 96),
}


def congestion_color(score: float) -> Tuple[int, int, int]:
    """Map 0..1 congestion score to green→red gradient."""
    r = int(clamp(score * 2, 0, 1) * 200 + 55)
    g = int(clamp(1 - score * 2 + 1, 0, 1) * 200 + 55)
    b = 60
    return (r, g, b)


def heatmap_color(usage: float, max_usage: float) -> Tuple[int, int, int]:
    """Map lane usage count to a heatmap color."""
    t = normalize(usage, 0, max(max_usage, 1))
    r = int(lerp(50, 255, t))
    g = int(lerp(200, 50, t))
    b = int(lerp(200, 50, t))
    return (r, g, b)
