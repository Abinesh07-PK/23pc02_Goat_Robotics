"""simulator.py — Main simulation loop, event manager, and robot execution engine."""

import logging
from collections import deque
from typing import Dict, List, Optional, Any

from core import MapGraph, Robot, Planner
from traffic import TrafficManager
from intelligence import CostEngine, PolicyEngine, PriorityEngine, FlowAllocator
from metrics import MetricsTracker, SimLogger, ReplayRecorder
from utils import EventBus, load_config, load_map, setup_logging

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# EVENT TYPES
# ══════════════════════════════════════════════════════════════

EVT_ROBOT_BLOCKED    = "ROBOT_BLOCKED"
EVT_GOAL_REACHED     = "GOAL_REACHED"
EVT_DEADLOCK         = "DEADLOCK_DETECTED"
EVT_CONGESTION_SPIKE = "CONGESTION_SPIKE"
EVT_REPLAN           = "REPLAN_TRIGGERED"
EVT_EMERGENCY_STOP   = "EMERGENCY_STOP"


# ══════════════════════════════════════════════════════════════
# SIMULATOR
# ══════════════════════════════════════════════════════════════

class Simulator:
    """
    Controls global simulation time, robot execution, and all module interaction.

    Core loop:
      1. Traffic update (reservations, congestion, conflicts, deadlocks)
      2. Intelligence (cost, policy, replan decisions)
      3. Robot motion step
      4. Metrics & events
      5. Render
    """

    def __init__(self, config_path: str = "config.yaml", map_path: str = "map.json"):
        self.config = load_config(config_path)
        self.map_data = load_map(map_path)

        sim_cfg  = self.config.get("simulation", {})
        log_cfg  = self.config.get("logging", {})
        setup_logging(log_cfg.get("log_file", "simulation.log"), log_cfg.get("log_level", "INFO"))

        # Time
        self.tick:          int   = 0
        self.delta_time:    float = sim_cfg.get("delta_time", 0.1)
        self.max_ticks:     int   = sim_cfg.get("max_time_steps", 10000)
        self.running:       bool  = False
        self.paused:        bool  = False

        # Graph
        self.graph = MapGraph(self.map_data)

        # Intelligence modules
        self.priority_engine = PriorityEngine(self.config.get("priority", {}))
        self.policy_engine   = PolicyEngine(self.config.get("planner", {}))
        self.cost_engine     = CostEngine(self.config.get("intelligence", {}))
        self.flow_allocator  = FlowAllocator(self.graph, update_interval=40)

        # Planner
        self.planner = Planner(self.graph, self.config.get("planner", {}))

        # Traffic
        self.traffic = TrafficManager(
            self.graph, self.config, self.priority_engine, self.policy_engine
        )

        # Robots
        self.robots: List[Robot] = []
        self._init_robots()

        # Metrics & logging
        self.metrics  = MetricsTracker()
        self.sim_log  = SimLogger(log_cfg.get("log_file", "simulation.log"))
        self.replay   = ReplayRecorder(max_snapshots=5000)

        # Event bus
        self.events = EventBus()
        self._register_event_handlers()

        # Visualizer reference (set externally)
        self.visualizer = None

        logger.info(f"Simulator initialized: {len(self.robots)} robots, "
                    f"{len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")

    # ── initialisation ────────────────────────────────────────

    def _init_robots(self):
        cfg = self.config.get("simulation", {})
        robot_cfgs = self.map_data.get("robot_configs", [])
        num = cfg.get("num_robots", len(robot_cfgs))

        for i, rc in enumerate(robot_cfgs[:num]):
            robot = Robot(
                robot_id=rc["id"],
                start_node=rc["start"],
                goal_node=rc["goal"],
                allowed_safety_max=rc.get("allowed_safety_max", 3),
                config=self.config,
            )
            # Set initial position
            node = self.graph.nodes.get(robot.start_node)
            if node:
                robot.x, robot.y = node.x, node.y
            self.robots.append(robot)

        # Initial planning for all robots
        cost_fn = self.cost_engine.make_cost_fn()
        for robot in self.robots:
            path = self.planner.plan(
                robot, self.traffic.reservation_manager.get_table(), cost_fn, self.tick
            )
            robot.set_path(path)
            if path:
                self.traffic.request_reservation(robot.id, path, self.tick)

    def _register_event_handlers(self):
        self.events.subscribe(EVT_GOAL_REACHED, self._on_goal_reached)
        self.events.subscribe(EVT_DEADLOCK, self._on_deadlock)
        self.events.subscribe(EVT_ROBOT_BLOCKED, self._on_robot_blocked)

    # ── event handlers ────────────────────────────────────────

    def _on_goal_reached(self, payload: Dict):
        robot = payload.get("robot")
        if robot is None:
            return
        logger.info(f"Robot {robot.id} reached goal! Total completions: {robot.goals_completed}")
        self.sim_log.log(EVT_GOAL_REACHED, {"robot_id": robot.id}, self.tick)
        # Assign new goal (cycle back)
        robot.goal_node, robot.start_node = robot.start_node, robot.goal_node
        robot.current_node = robot.path[-1][0] if robot.path else robot.current_node
        robot.path = []
        robot.path_index = 0
        robot.wait_ticks = 0
        robot.stuck_ticks = 0
        robot.state = "IDLE"
        self._plan_robot(robot)

    def _on_deadlock(self, payload: Dict):
        self.metrics.record_deadlock(detected=True)
        self.sim_log.log(EVT_DEADLOCK, payload, self.tick)

    def _on_robot_blocked(self, payload: Dict):
        self.sim_log.log(EVT_ROBOT_BLOCKED, payload, self.tick)

    # ── main loop ─────────────────────────────────────────────

    def run(self, headless: bool = False):
        """Run the simulation loop."""
        self.running = True
        logger.info("Simulation started.")

        while self.running and self.tick < self.max_ticks:
            # 1. Always handle Pygame events so window never freezes
            if not headless and self.visualizer:
                if not self.visualizer.handle_events(self):
                    break

            # 2. Skip logic tick if paused, but still redraw frame
            if self.paused:
                if not headless and self.visualizer:
                    self.visualizer.render(self.graph, self.robots, self.tick, self.metrics)
                continue

            # 3. Main Logic Step
            self._step()

            # 4. Render New Frame
            if not headless and self.visualizer:
                self.visualizer.render(
                    self.graph, self.robots, self.tick, self.metrics
                )

            self.tick += 1

        self._finalise()

    def step_once(self):
        """Advance simulation by one tick (for external control / pygame loop)."""
        if self.running and not self.paused:
            self._step()
            self.tick += 1
        return self.running

    def _step(self):
        """One complete simulation tick."""

        # 1. Update cost engine's max_usage for normalisation
        self.cost_engine.update_max_usage(self.graph.max_usage_count())
        cost_fn = self.cost_engine.make_cost_fn()

        # 2. Traffic module: reservations, congestion, conflicts, deadlocks
        system_state = self.traffic.update(self.robots, self.tick, self.planner, cost_fn)

        # Publish deadlock events
        if system_state.get("deadlock_participants"):
            self.events.publish(EVT_DEADLOCK, {
                "participants": list(system_state["deadlock_participants"])
            })

        # 3. Flow allocator macro-routing update
        self.flow_allocator.update(self.robots, self.tick)

        # 4. Intelligence: decide replan for each robot
        for robot in self.robots:
            if robot.state in ("GOAL_REACHED",):
                continue

            should, reason = self.policy_engine.should_replan(robot, system_state, self.tick)
            if should:
                new_path = self.planner.replan(robot, reason, 
                                                self.traffic.reservation_manager.get_table(),
                                                cost_fn, self.tick)
                robot.set_path(new_path)
                if new_path:
                    self.traffic.request_reservation(robot.id, new_path, self.tick)
                    robot.state = "MOVING"
                    self.events.publish(EVT_REPLAN, {"robot_id": robot.id, "reason": reason})
                else:
                    robot.state = "BLOCKED"

        # 5. Execute robot motion
        for robot in self.robots:
            self._execute_robot(robot, cost_fn)

        # 6. Metrics tick
        self.metrics.tick(self.robots, self.graph, self.tick)
        self.replay.record(self.robots, self.graph, self.tick)

        # 7. Flush events
        self.events.flush()

    # ── robot motion execution ────────────────────────────────

    def _execute_robot(self, robot: Robot, cost_fn):
        """Step one robot forward in its current state."""

        if robot.state == "GOAL_REACHED":
            return

        if robot.state == "EMERGENCY_STOP":
            # Attempt recovery after a few ticks
            robot.wait_ticks += 1
            if robot.wait_ticks > 20:
                robot.state = "IDLE"
                robot.wait_ticks = 0
            return

        if robot.state == "IDLE":
            if not robot.path:
                self._plan_robot(robot)
            if robot.path:
                robot.state = "MOVING"
            return

        if robot.state == "WAITING":
            robot.wait_ticks += 1
            robot.total_ticks += 1
            # After threshold: switch to MOVING to try again
            if robot.wait_ticks > self.policy_engine.wait_delay_threshold:
                robot.state = "MOVING"
                robot.stuck_ticks += 1
            return

        if robot.state in ("MOVING", "REPLANNING"):
            self._move_robot(robot)

        if robot.state == "BLOCKED":
            robot.wait_ticks += 1
            robot.stuck_ticks += 1
            robot.total_ticks += 1

    def _move_robot(self, robot: Robot):
        """Attempt to advance robot along its path."""
        if not robot.path or robot.path_index >= len(robot.path):
            if robot.current_node == robot.goal_node:
                robot.state = "GOAL_REACHED"
                robot.goals_completed += 1
                self.events.publish(EVT_GOAL_REACHED, {"robot": robot})
            else:
                robot.state = "IDLE"
            return

        next_node, next_eid = robot.path[robot.path_index]
        edge = self.graph.get_edge(next_eid)
        if edge is None:
            robot.path_index += 1
            return

        direction = "rev" if next_eid.endswith("_rev") else "fwd"

        # Safety check
        if edge.safety_level > robot.allowed_safety_max:
            robot.state = "BLOCKED"
            return

        # Safe following distance check
        if not robot.enforce_safe_distance(self.graph, self.robots):
            robot.state = "WAITING"
            robot.wait_ticks += 1
            self.events.publish(EVT_ROBOT_BLOCKED, {
                "robot_id": robot.id, "reason": "safe_distance"
            })
            return

        # Capacity check
        if not edge.can_enter(robot.id, direction):
            robot.state = "WAITING"
            robot.wait_ticks += 1
            return

        # Enter edge
        if robot.current_edge_id != next_eid:
            # Exit previous edge
            if robot.current_edge_id:
                prev_edge = self.graph.get_edge(robot.current_edge_id)
                if prev_edge:
                    prev_dir = "rev" if robot.current_edge_id.endswith("_rev") else "fwd"
                    prev_edge.exit(robot.id, prev_dir)
            edge.enter(robot.id, direction)
            robot.current_edge_id = next_eid
            robot.edge_direction = direction
            robot.progress = 0.0

        # Determine speed (lane max speed, reduced by congestion)
        effective_speed = edge.max_speed * (1.0 - 0.5 * edge.congestion_score)
        effective_speed = max(effective_speed, 0.1)

        # Advance progress
        traversal_time = edge.length / effective_speed
        progress_per_tick = self.delta_time / traversal_time
        robot.progress = min(robot.progress + progress_per_tick, 1.0)

        # Update world position
        robot.update_position(self.graph, self.delta_time)

        # Check if edge traversal complete
        if robot.progress >= 1.0:
            edge.exit(robot.id, direction)
            robot.current_edge_id = None
            robot.advance_path()
            robot.wait_ticks = 0
            robot.stuck_ticks = 0
            robot.state = "MOVING"

        robot.total_ticks += 1

    # ── helpers ───────────────────────────────────────────────

    def _plan_robot(self, robot: Robot):
        cost_fn = self.cost_engine.make_cost_fn()
        path = self.planner.plan(
            robot, self.traffic.reservation_manager.get_table(), cost_fn, self.tick
        )
        robot.set_path(path)
        if path:
            self.traffic.request_reservation(robot.id, path, self.tick)

    def _finalise(self):
        self.running = False
        self.replay.stop()

        # Clean up edge occupancies
        for robot in self.robots:
            if robot.current_edge_id:
                e = self.graph.get_edge(robot.current_edge_id)
                if e:
                    e.exit(robot.id, robot.edge_direction)

        self.metrics.print_summary()
        metrics_path = self.config.get("logging", {}).get("metrics_file", "metrics.json")
        self.metrics.save(metrics_path)
        self.metrics.export_markdown_summary("RESULTS_SUMMARY.md")
        self.replay.save("replay.json")
        self.sim_log.flush()
        logger.info("Simulation complete.")

    def pause(self):
        self.paused = not self.paused
        logger.info(f"Simulation {'paused' if self.paused else 'resumed'}.")

    def stop(self):
        self.running = False
