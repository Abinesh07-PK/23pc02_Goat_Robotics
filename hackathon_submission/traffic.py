"""traffic.py — Reservation manager, congestion engine, coordination, deadlock detection & resolution."""

import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple, Any

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# RESERVATION MANAGER
# ══════════════════════════════════════════════════════════════

class ReservationManager:
    """
    Time-space reservation system.
    reservation_table[(edge_id, time_step)] = robot_id
    """

    def __init__(self, reservation_horizon: int = 20):
        self.horizon = reservation_horizon
        # (edge_id, time_step) → robot_id
        self._table: Dict[Tuple[str, int], str] = {}
        # robot_id → set of (edge_id, time_step) keys it owns
        self._robot_reservations: Dict[str, Set[Tuple]] = defaultdict(set)

    # ── core operations ───────────────────────────────────────

    def is_available(self, edge_id: str, time_step: int, robot_id: str) -> bool:
        key = (edge_id, time_step)
        holder = self._table.get(key)
        return holder is None or holder == robot_id

    def reserve_path(self, robot_id: str, path: List[Tuple[str, str]], current_time: int,
                     graph=None) -> bool:
        """
        Attempt to reserve all edges in path.
        path: [(next_node, edge_id), ...]
        Returns True if fully reserved, False if any conflict found.
        """
        proposed: List[Tuple] = []
        for i, (node, edge_id) in enumerate(path):
            t = current_time + i + 1
            if t > current_time + self.horizon:
                break
            base_eid = edge_id.replace("_rev", "")

            # Check critical lane — must be free
            if graph:
                e = graph.get_edge(edge_id)
                if e and e.is_critical:
                    if not self.is_available(base_eid, t, robot_id):
                        return False

            if not self.is_available(base_eid, t, robot_id):
                return False
            proposed.append((base_eid, t))

        # Commit
        self.release_path(robot_id)
        for base_eid, t in proposed:
            key = (base_eid, t)
            self._table[key] = robot_id
            self._robot_reservations[robot_id].add(key)
        return True

    def release_path(self, robot_id: str):
        for key in self._robot_reservations.pop(robot_id, set()):
            if self._table.get(key) == robot_id:
                del self._table[key]

    def get_conflicts(self, robot_id: str, path: List[Tuple[str, str]],
                      current_time: int) -> List[str]:
        """Returns list of edge_ids that have conflicts."""
        conflicts = []
        for i, (node, edge_id) in enumerate(path):
            t = current_time + i + 1
            base_eid = edge_id.replace("_rev", "")
            holder = self._table.get((base_eid, t))
            if holder and holder != robot_id:
                conflicts.append(edge_id)
        return conflicts

    def purge_old(self, current_time: int):
        """Remove reservations older than current_time to keep table compact."""
        stale = [k for k in self._table if k[1] < current_time - 2]
        for k in stale:
            rid = self._table.pop(k)
            self._robot_reservations[rid].discard(k)

    def get_table(self) -> Dict:
        return self._table

    def who_holds(self, edge_id: str, time_step: int) -> Optional[str]:
        return self._table.get((edge_id.replace("_rev", ""), time_step))


# ══════════════════════════════════════════════════════════════
# CONGESTION ENGINE
# ══════════════════════════════════════════════════════════════

class CongestionEngine:
    """Maintains dynamic congestion scores and detects hotspots."""

    def __init__(self, hotspot_threshold: float = 0.75):
        self.hotspot_threshold = hotspot_threshold
        self.hotspots: Set[str] = set()    # edge_ids currently hotspots

    def update(self, graph, max_usage: float = 1.0):
        """Recompute congestion for all edges and update hotspot set."""
        self.hotspots.clear()
        graph.update_all_congestion(max_usage=max_usage)
        seen = set()
        for eid, edge in graph.edges.items():
            if edge.id in seen:
                continue
            seen.add(edge.id)
            if edge.congestion_score >= self.hotspot_threshold:
                self.hotspots.add(edge.id)

    def is_hotspot(self, edge_id: str) -> bool:
        return edge_id.replace("_rev", "") in self.hotspots

    def get_score(self, edge_id: str, graph) -> float:
        edge = graph.get_edge(edge_id)
        return edge.congestion_score if edge else 0.0


# ══════════════════════════════════════════════════════════════
# DEADLOCK DETECTOR  (Wait-For Graph + DFS cycle detection)
# ══════════════════════════════════════════════════════════════

class DeadlockDetector:
    """
    Builds a Wait-For Graph (WFG) among robots and detects cycles.
    Node = robot_id; Edge A→B = "Robot A is waiting for Robot B".
    """

    def build_wfg(self, robots: List, reservation_table: Dict,
                  graph) -> Dict[str, List[str]]:
        """Returns adjacency dict: robot_id → [robot_ids it waits for]."""
        wfg: Dict[str, List[str]] = {r.id: [] for r in robots}
        robot_map = {r.id: r for r in robots}

        for robot in robots:
            if robot.state not in ("WAITING", "BLOCKED"):
                continue
            next_eid = robot.peek_next_edge()
            if next_eid is None:
                continue
            base_eid = next_eid.replace("_rev", "")
            # Find which robot holds the next time slot
            # Check upcoming 3 time steps
            for dt in range(1, 4):
                holder_id = reservation_table.get((base_eid, robot.total_ticks + dt))
                if holder_id and holder_id != robot.id and holder_id in robot_map:
                    wfg[robot.id].append(holder_id)
                    break
        return wfg

    def detect_cycles(self, wfg: Dict[str, List[str]]) -> List[List[str]]:
        """Returns list of cycles, each cycle is a list of robot_ids."""
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        cycles: List[List[str]] = []

        def dfs(node: str, path: List[str]):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            for neighbor in wfg.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, path)
                elif neighbor in rec_stack:
                    # Found a cycle — extract it
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:])
            path.pop()
            rec_stack.discard(node)

        for node in wfg:
            if node not in visited:
                dfs(node, [])

        return cycles

    def get_deadlock_groups(self, robots: List, reservation_table: Dict,
                             graph) -> Tuple[List[List[str]], Set[str]]:
        """
        Returns (cycles, participant_set).
        cycles: list of deadlocked robot groups.
        participant_set: flat set of all deadlocked robot_ids.
        """
        wfg = self.build_wfg(robots, reservation_table, graph)
        cycles = self.detect_cycles(wfg)
        participants: Set[str] = set()
        for cycle in cycles:
            participants.update(cycle)
        return cycles, participants


# ══════════════════════════════════════════════════════════════
# DEADLOCK RESOLVER
# ══════════════════════════════════════════════════════════════

class DeadlockResolver:
    """
    Breaks deadlocks by choosing a victim robot and applying a resolution strategy.
    """

    def __init__(self, priority_engine):
        self.priority_engine = priority_engine

    def resolve(self, cycle: List[str], robot_map: Dict, graph,
                reservation_manager: "ReservationManager") -> str:
        """
        Resolves one cycle. Returns the victim robot_id.
        Strategy order:
          1. Force-wait the lowest-priority robot briefly
          2. If that fails → force reroute
          3. If safety-critical edge involved → emergency stop all
        """
        robots_in_cycle = [robot_map[rid] for rid in cycle if rid in robot_map]
        if not robots_in_cycle:
            return ""

        # Check if any robot is on a high-safety lane
        high_safety = any(
            graph.get_edge(r.current_edge_id) and
            graph.get_edge(r.current_edge_id).safety_level >= 5
            for r in robots_in_cycle if r.current_edge_id
        )
        if high_safety:
            for r in robots_in_cycle:
                r.emergency_stop("deadlock in high-safety zone")
            logger.warning(f"Emergency stop for cycle: {cycle}")
            return ""

        victim = self.priority_engine.lowest_priority(robots_in_cycle)
        if victim is None:
            return ""

        # Strategy 1: force short wait
        victim.state = "WAITING"
        victim.wait_ticks = 0
        reservation_manager.release_path(victim.id)

        logger.info(f"DeadlockResolver: victim={victim.id} in cycle {cycle} → FORCE WAIT")
        return victim.id


# ══════════════════════════════════════════════════════════════
# COORDINATION MANAGER  (priority arbitration, PIBT-lite)
# ══════════════════════════════════════════════════════════════

class CoordinationManager:
    """
    Handles conflict resolution using priority ordering.
    Implements lightweight priority inheritance.
    """

    def __init__(self, priority_engine):
        self.priority_engine = priority_engine
        # Temporary priority boosts: robot_id → boost_ticks_remaining
        self._priority_boosts: Dict[str, int] = {}

    def resolve_conflict(self, robots_in_conflict: List, graph,
                          reservation_manager: "ReservationManager",
                          policy_engine, system_state: Dict,
                          current_tick: int) -> Dict[str, str]:
        """
        Given a list of robots competing for the same resource, determine winner/losers.
        Returns {robot_id: "WIN"|"WAIT"|"REROUTE"} decisions.
        """
        decisions: Dict[str, str] = {}
        ranked = self.priority_engine.rank(robots_in_conflict)

        winner = ranked[0]
        decisions[winner.id] = "WIN"

        for loser in ranked[1:]:
            action = policy_engine.decide_wait_or_reroute(loser, system_state)
            decisions[loser.id] = action
            if action == "WAIT":
                loser.state = "WAITING"
                # Priority inheritance: loser temporarily inherits winner's priority
                self._inherit_priority(loser, winner)

        return decisions

    def _inherit_priority(self, blocked_robot, blocker_robot, ticks: int = 10):
        """
        blocked_robot inherits blocker's priority temporarily.
        """
        if blocker_robot.priority_score > blocked_robot.priority_score:
            self._priority_boosts[blocked_robot.id] = ticks
            blocked_robot.priority_score = blocker_robot.priority_score
            logger.debug(f"Priority inheritance: {blocked_robot.id} ← {blocker_robot.id}")

    def tick_boosts(self):
        """Decay priority boosts each tick."""
        expired = [rid for rid, ticks in self._priority_boosts.items() if ticks <= 0]
        for rid in expired:
            del self._priority_boosts[rid]
        for rid in self._priority_boosts:
            self._priority_boosts[rid] -= 1


# ══════════════════════════════════════════════════════════════
# TRAFFIC MANAGER  (central authority)
# ══════════════════════════════════════════════════════════════

class TrafficManager:
    """
    Single source of truth coordinating all traffic subsystems.

    Orchestrates:
      - ReservationManager
      - CongestionEngine
      - CoordinationManager
      - DeadlockDetector
      - DeadlockResolver
    """

    def __init__(self, graph, config: Dict, priority_engine, policy_engine):
        cfg_t = config.get("traffic", {})
        cfg_p = config.get("planner", {})

        self.graph = graph
        self.reservation_manager = ReservationManager(
            reservation_horizon=cfg_p.get("reservation_horizon", 20)
        )
        self.congestion_engine = CongestionEngine(
            hotspot_threshold=cfg_t.get("congestion_hotspot_threshold", 0.75)
        )
        self.coordination_manager = CoordinationManager(priority_engine)
        self.deadlock_detector = DeadlockDetector()
        self.deadlock_resolver = DeadlockResolver(priority_engine)

        self.policy_engine = policy_engine
        self.deadlock_check_interval: int = cfg_t.get("deadlock_check_interval", 5)

        # Shared state snapshot (published each tick, consumed by intelligence)
        self.system_state: Dict = {
            "reservation_denied": {},
            "congestion_hotspots": set(),
            "deadlock_participants": set(),
            "next_edge_safety": {},
        }

    # ── main update loop ──────────────────────────────────────

    def update(self, robots: List, current_tick: int, planner, cost_fn) -> Dict:
        """
        Called once per simulation tick.
        Returns updated system_state for intelligence module.
        """
        self._update_congestion()
        self._process_reservations(robots, current_tick, planner, cost_fn)
        self._resolve_conflicts(robots, current_tick)

        if current_tick % self.deadlock_check_interval == 0:
            self._detect_and_resolve_deadlocks(robots)

        self.coordination_manager.tick_boosts()
        self.reservation_manager.purge_old(current_tick)

        # Build next_edge_safety map
        for robot in robots:
            next_eid = robot.peek_next_edge()
            if next_eid:
                e = self.graph.get_edge(next_eid)
                self.system_state["next_edge_safety"][robot.id] = e.safety_level if e else 1

        return self.system_state

    # ── reservation processing ────────────────────────────────

    def _process_reservations(self, robots: List, current_tick: int, planner, cost_fn):
        denied: Dict[str, bool] = {}
        for robot in robots:
            if robot.state in ("GOAL_REACHED", "EMERGENCY_STOP"):
                continue
            if not robot.path:
                continue
            remaining_path = robot.path[robot.path_index:]
            success = self.reservation_manager.reserve_path(
                robot.id, remaining_path, current_tick, graph=self.graph
            )
            if not success:
                denied[robot.id] = True
                robot.state = "BLOCKED" if robot.state == "MOVING" else robot.state
        self.system_state["reservation_denied"] = denied

    # ── congestion update ─────────────────────────────────────

    def _update_congestion(self):
        max_usage = self.graph.max_usage_count()
        self.congestion_engine.update(self.graph, max_usage=max_usage)
        self.system_state["congestion_hotspots"] = set(self.congestion_engine.hotspots)

    # ── conflict resolution ───────────────────────────────────

    def _resolve_conflicts(self, robots: List, current_tick: int):
        """
        Group robots competing for the same next edge and arbitrate.
        """
        # Build map: (next_edge_id_base) → list of robots trying to enter
        edge_contenders: Dict[str, List] = defaultdict(list)
        for robot in robots:
            if robot.state in ("GOAL_REACHED", "EMERGENCY_STOP", "WAITING"):
                continue
            next_eid = robot.peek_next_edge()
            if next_eid:
                base = next_eid.replace("_rev", "")
                edge_contenders[base].append(robot)

        for edge_id, contenders in edge_contenders.items():
            edge = self.graph.get_edge(edge_id)
            if edge is None:
                continue
            # Only conflict if more contenders than capacity
            if len(contenders) > edge.capacity:
                self.coordination_manager.resolve_conflict(
                    contenders, self.graph,
                    self.reservation_manager,
                    self.policy_engine,
                    self.system_state,
                    current_tick
                )

    # ── deadlock detection & resolution ──────────────────────

    def _detect_and_resolve_deadlocks(self, robots: List):
        reservation_table = self.reservation_manager.get_table()
        cycles, participants = self.deadlock_detector.get_deadlock_groups(
            robots, reservation_table, self.graph
        )
        self.system_state["deadlock_participants"] = participants

        if cycles:
            logger.warning(f"Deadlocks detected: {len(cycles)} cycle(s), participants={participants}")
            robot_map = {r.id: r for r in robots}
            for cycle in cycles:
                self.deadlock_resolver.resolve(
                    cycle, robot_map, self.graph, self.reservation_manager
                )

    # ── external interface ────────────────────────────────────

    def request_reservation(self, robot_id: str, path: List[Tuple[str, str]],
                             current_time: int) -> bool:
        return self.reservation_manager.reserve_path(robot_id, path, current_time, self.graph)

    def release_reservation(self, robot_id: str):
        self.reservation_manager.release_path(robot_id)

    def get_congestion_score(self, edge_id: str) -> float:
        return self.congestion_engine.get_score(edge_id, self.graph)
