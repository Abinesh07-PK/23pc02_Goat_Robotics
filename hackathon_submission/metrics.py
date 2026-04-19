"""metrics.py — Structured logging, performance metrics, and optional replay."""

import json
import time
import logging
from collections import defaultdict
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# EVENT LOGGER
# ══════════════════════════════════════════════════════════════

class SimLogger:
    """Structured JSON event log."""

    def __init__(self, log_file: str = "simulation.log"):
        self.log_file = log_file
        self._entries: List[Dict] = []

    def log(self, event_type: str, payload: Any = None, tick: int = 0):
        entry = {"tick": tick, "event": event_type, "data": payload}
        self._entries.append(entry)
        logger.debug(f"[{tick}] {event_type}: {payload}")

    def flush(self, path: Optional[str] = None):
        out = path or self.log_file
        try:
            with open(out, "w") as f:
                json.dump(self._entries, f, indent=2, default=str)
            logger.info(f"Simulation log saved → {out}")
        except Exception as e:
            logger.error(f"Failed to write log: {e}")

    def get_entries(self) -> List[Dict]:
        return list(self._entries)


# ══════════════════════════════════════════════════════════════
# METRICS TRACKER
# ══════════════════════════════════════════════════════════════

class MetricsTracker:
    """
    Tracks per-robot and global simulation metrics.
    Computes throughput, average delay, lane utilization, deadlock counts.
    """

    def __init__(self):
        # Per-robot
        self.robot_wait_ticks:     Dict[str, int]   = defaultdict(int)
        self.robot_move_ticks:     Dict[str, int]   = defaultdict(int)
        self.robot_goals:          Dict[str, int]   = defaultdict(int)
        self.robot_replans:        Dict[str, int]   = defaultdict(int)
        self.robot_stuck_events:   Dict[str, int]   = defaultdict(int)
        self.robot_emergency_stops: Dict[str, int]  = defaultdict(int)

        # Global
        self.deadlocks_detected:   int = 0
        self.deadlocks_resolved:   int = 0
        self.total_ticks:          int = 0
        self.start_real_time:      float = time.time()

        # Lane utilization: edge_id → tick count occupied
        self.lane_ticks:           Dict[str, int]   = defaultdict(int)

        # Snapshot history for live display
        self._throughput_history:  List[float] = []
        self._congestion_history:  List[float] = []

    def tick(self, robots: List, graph, tick: int):
        """Update metrics for current tick."""
        self.total_ticks = tick
        active_moving = 0

        for robot in robots:
            if robot.state == "MOVING":
                self.robot_move_ticks[robot.id] += 1
                active_moving += 1
            elif robot.state in ("WAITING", "BLOCKED"):
                self.robot_wait_ticks[robot.id] += 1
            elif robot.state == "EMERGENCY_STOP":
                self.robot_emergency_stops[robot.id] += 1

            self.robot_goals[robot.id]   = robot.goals_completed
            self.robot_replans[robot.id] = robot.replan_count

            # Lane occupancy
            if robot.current_edge_id:
                base = robot.current_edge_id.replace("_rev", "")
                self.lane_ticks[base] += 1

        # Throughput snapshot (goals per 100 ticks)
        if tick % 100 == 0 and tick > 0:
            total_goals = sum(self.robot_goals.values())
            tp = total_goals / (tick / 100)
            self._throughput_history.append(tp)

        # Mean congestion snapshot
        if tick % 50 == 0 and tick > 0:
            seen = set()
            scores = []
            for eid, edge in graph.edges.items():
                if edge.id not in seen:
                    scores.append(edge.congestion_score)
                    seen.add(edge.id)
            if scores:
                self._congestion_history.append(sum(scores) / len(scores))

    def record_deadlock(self, detected: bool = True):
        if detected:
            self.deadlocks_detected += 1
        else:
            self.deadlocks_resolved += 1

    def summary(self) -> Dict:
        elapsed = time.time() - self.start_real_time
        total_goals = sum(self.robot_goals.values())
        total_waits = sum(self.robot_wait_ticks.values())
        total_moves = sum(self.robot_move_ticks.values())
        total_replans = sum(self.robot_replans.values())

        avg_wait = total_waits / max(len(self.robot_wait_ticks), 1)
        throughput = total_goals / max(self.total_ticks / 100, 1)

        # Lane utilization: fraction of ticks each lane was occupied
        lane_util = {
            eid: ticks / max(self.total_ticks, 1)
            for eid, ticks in self.lane_ticks.items()
        }

        return {
            "total_ticks": self.total_ticks,
            "elapsed_seconds": round(elapsed, 2),
            "total_goals_completed": total_goals,
            "total_replans": total_replans,
            "throughput_per_100_ticks": round(throughput, 3),
            "average_wait_ticks": round(avg_wait, 2),
            "deadlocks_detected": self.deadlocks_detected,
            "deadlocks_resolved": self.deadlocks_resolved,
            "per_robot": {
                rid: {
                    "goals": self.robot_goals[rid],
                    "wait_ticks": self.robot_wait_ticks[rid],
                    "move_ticks": self.robot_move_ticks[rid],
                    "replans": self.robot_replans[rid],
                    "emergency_stops": self.robot_emergency_stops[rid],
                }
                for rid in self.robot_goals
            },
            "lane_utilization": {k: round(v, 4) for k, v in lane_util.items()},
            "throughput_history": self._throughput_history,
            "mean_congestion_history": self._congestion_history,
        }

    def save(self, path: str = "metrics.json"):
        data = self.summary()
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Metrics saved → {path}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    def print_summary(self):
        s = self.summary()
        print("\n" + "═" * 50)
        print("  SIMULATION METRICS SUMMARY")
        print("═" * 50)
        print(f"  Ticks: {s['total_ticks']}  |  Real time: {s['elapsed_seconds']}s")
        print(f"  Goals completed:  {s['total_goals_completed']}")
        print(f"  Throughput:       {s['throughput_per_100_ticks']} goals/100 ticks")
        print(f"  Avg wait:         {s['average_wait_ticks']} ticks")
        print(f"  Total replans:    {s['total_replans']}")
        print(f"  Deadlocks:        detected={s['deadlocks_detected']}  resolved={s['deadlocks_resolved']}")
        print("─" * 50)
        for rid, stats in s["per_robot"].items():
            print(f"  {rid}: goals={stats['goals']}  waits={stats['wait_ticks']}  replans={stats['replans']}")
        print("═" * 50 + "\n")

    def export_markdown_summary(self, path: str = "RESULTS_SUMMARY.md"):
        s = self.summary()
        content = f"""# Simulation Results Summary

## Overview
- **Total Ticks**: {s['total_ticks']}
- **Real-Time Elapsed**: {s['elapsed_seconds']} s

## Performance Metrics
- **Total Goals Completed**: {s['total_goals_completed']}
- **System Throughput**: {s['throughput_per_100_ticks']} goals / 100 ticks
- **Average Wait Time**: {s['average_wait_ticks']} ticks
- **Path Re-planning Events**: {s['total_replans']}

## Deadlock Analysis
- **Deadlocks Detected**: {s['deadlocks_detected']}
- **Deadlocks Resolved**: {s['deadlocks_resolved']}

## Fleet Overview
| Robot ID | Goals | Wait Ticks | Move Ticks | Replans | Emergency Stops |
|---|---|---|---|---|---|
"""
        for rid, stats in s["per_robot"].items():
            content += f"| {rid} | {stats['goals']} | {stats['wait_ticks']} | {stats['move_ticks']} | {stats['replans']} | {stats['emergency_stops']} |\n"
            
        try:
            with open(path, "w") as f:
                f.write(content)
            logger.info(f"Markdown summary saved → {path}")
        except Exception as e:
            logger.error(f"Failed to save markdown summary: {e}")


# ══════════════════════════════════════════════════════════════
# REPLAY RECORDER (optional — great for demo)
# ══════════════════════════════════════════════════════════════

class ReplayRecorder:
    """
    Records a lightweight snapshot each tick for post-hoc replay.
    Snapshots are written to a JSON file on completion.
    """

    def __init__(self, max_snapshots: int = 5000):
        self.max_snapshots = max_snapshots
        self._snapshots: List[Dict] = []
        self._recording = True

    def record(self, robots: List, graph, tick: int):
        if not self._recording or len(self._snapshots) >= self.max_snapshots:
            return

        robot_states = []
        for r in robots:
            robot_states.append({
                "id": r.id,
                "state": r.state,
                "node": r.current_node,
                "x": round(r.x, 1),
                "y": round(r.y, 1),
                "edge": r.current_edge_id,
                "progress": round(r.progress, 3),
            })

        edge_states = []
        seen = set()
        for eid, edge in graph.edges.items():
            if edge.id in seen:
                continue
            seen.add(edge.id)
            edge_states.append({
                "id": edge.id,
                "occ": edge.current_occupancy,
                "cong": round(edge.congestion_score, 3),
                "usage": edge.usage_count,
            })

        self._snapshots.append({
            "tick": tick,
            "robots": robot_states,
            "edges": edge_states,
        })

    def stop(self):
        self._recording = False

    def save(self, path: str = "replay.json"):
        try:
            with open(path, "w") as f:
                json.dump(self._snapshots, f)
            logger.info(f"Replay saved → {path} ({len(self._snapshots)} snapshots)")
        except Exception as e:
            logger.error(f"Failed to save replay: {e}")

    def get_snapshots(self) -> List[Dict]:
        return self._snapshots
