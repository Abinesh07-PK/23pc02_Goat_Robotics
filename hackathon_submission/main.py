"""main.py — Entry point for the Lane-Aware Multi-Robot Traffic Control System."""

import argparse
import sys
import os
import logging

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from simulator import Simulator
from visualizer import Visualizer, PYGAME_AVAILABLE
from utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Lane-Aware Multi-Robot Traffic Control — GOAT Hackathon"
    )
    p.add_argument("--config",   default="config.yaml",  help="Path to config.yaml")
    p.add_argument("--map",      default="map.json",      help="Path to map.json")
    p.add_argument("--headless", action="store_true",     help="Run without pygame visualizer")
    p.add_argument("--ticks",    type=int, default=None,  help="Override max simulation ticks")
    p.add_argument("--fps",      type=int, default=None,  help="Override visualizer FPS")
    return p.parse_args()


def main():
    args = parse_args()

    setup_logging()
    logger.info("=" * 60)
    logger.info("  Lane-Aware Multi-Robot Traffic Control")
    logger.info("  GOAT Hackathon")
    logger.info("=" * 60)

    # Build simulator
    sim = Simulator(config_path=args.config, map_path=args.map)

    # Override ticks if specified
    if args.ticks:
        sim.max_ticks = args.ticks

    # Headless mode: no pygame
    headless = args.headless or not PYGAME_AVAILABLE

    if not headless:
        vis_cfg = sim.config.get("visualization", {})
        if args.fps:
            vis_cfg["fps"] = args.fps
        visualizer = Visualizer(sim.config)

        if visualizer.is_active():
            sim.visualizer = visualizer
            try:
                sim.run(headless=False)
            finally:
                visualizer.quit()
        else:
            logger.warning("Visualizer not available — falling back to headless mode.")
            sim.run(headless=True)
    else:
        logger.info("Running in headless mode.")
        _headless_run(sim)

    logger.info("Done.")


def _headless_run(sim: Simulator):
    """Headless loop with periodic console progress."""
    sim.running = True
    report_interval = max(sim.max_ticks // 20, 100)

    while sim.running and sim.tick < sim.max_ticks:
        sim.step_once()

        if sim.tick % report_interval == 0:
            s = sim.metrics.summary()
            active = sum(1 for r in sim.robots if r.state not in ("GOAL_REACHED", "EMERGENCY_STOP"))
            print(f"[Tick {sim.tick:5d}] "
                  f"Goals={s['total_goals_completed']:3d}  "
                  f"ActiveRobots={active}  "
                  f"Deadlocks={s['deadlocks_detected']}  "
                  f"Replans={s['total_replans']}")

        # Check if all robots are done
        if all(r.state == "GOAL_REACHED" for r in sim.robots):
            logger.info("All robots reached their goals.")
            break

    sim._finalise()


if __name__ == "__main__":
    main()
