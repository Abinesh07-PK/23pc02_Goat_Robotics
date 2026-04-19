# Lane-Aware Multi-Robot Traffic Control System 🚀

A highly-efficient, cooperative swarm-intelligence and traffic management system designed for structured environments (warehouses and factories). Built for the GOAT Hackathon.

## Features
- **Continuous Safe Following Distance**: True Euclidean constraint enforcing safe following distance dynamically on all edges.
- **Cooperative A* Intelligence**: Uses a Time-Space reservation table to route robots.
- **Micro-Congestion Heatmapping**: Robots proactively route around jammed factory corridors using visual tracking.
- **Deadlock Resolution**: Graph-based Wait-For-Graph engine detects and resolves cyclic wait conditions instantly through Priority Inheritance.
- **Dynamic Replanning**: On-the-fly traffic adjustments.

## System Architecture

### 1. `TrafficManager` (traffic.py)
The central authority. It maintains the global Time-Space `ReservationManager` where robots claim future time slots on specific edges, ensuring no two robots cross paths at the same tick.

### 2. `CostEngine` (intelligence.py)
When calculating A* paths, cost isn't just distance. The CostEngine dynamically generates edge weights using:
`wd*distance + α*congestion + β*safety_level + γ*contraflow + δ*historical_heatmap`

### 3. `PolicyEngine` (intelligence.py)
Determines when a robot should just sit and wait out traffic versus when it has been waiting too long (`wait_delay_threshold`) and needs to massively detour its route.

## How to Run

### Installation
Ensure Python 3.10+ is installed.
```bash
pip install pygame pyyaml
```

### Visual Simulation
To run the live simulation with the Pygame HUD:
```bash
python main.py
```
**Controls during simulation:**
- Press `[H]` to toggle Historical Heatmap View
- Press `[C]` to toggle Real-time Congestion Array View
- Press `[SPACE]` to pause/play.

### Headless Mode (Fast Metrics Gathering)
To run without Pygame and instantly generate the final `RESULTS_SUMMARY.md`:
```bash
python main.py --headless
```

## Hackathon Deliverables Tracker
- **Source Code**: Fully modularized (`core`, `intelligence`, `simulator`, `traffic`).
- **System Documentation**: Included in this README!
- **Video Demonstration**: *(Must be recorded by team using screen capture while running Visual Simulation)*
- **Results Summary**: Generated automatically at the end of the simulation as `RESULTS_SUMMARY.md`.
