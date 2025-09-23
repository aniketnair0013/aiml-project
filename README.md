# Autonomous Delivery Agent: Grid-Based Path Planning

An AI-powered delivery agent that navigates grid-based maps using informed and uninformed search algorithms, including support for dynamic obstacles (e.g., moving vehicles) with real-time replanning. Built in Python, it visualizes optimal paths on heatmaps, computes metrics like cost and efficiency, and includes unit tests for reliability.

This project demonstrates core AI concepts: pathfinding in static/dynamic environments, heuristic search (A*), stochastic optimization (Simulated Annealing), and simulation of real-world delivery scenarios.


## Features
- **Search Planners**:
  - BFS (Breadth-First Search): Complete and optimal for unweighted grids.
  - UCS (Uniform Cost Search): Handles varying terrain costs (e.g., rough vs. smooth paths).
  - A* (A-Star): Heuristic-based for efficient, near-optimal paths using Manhattan distance.
  - SA (Simulated Annealing): Stochastic method for escaping local optima, useful in noisy environments.
- **Map Support**:
  - Static maps: Small (5x5), medium (20x20), large (50x50) grids with obstacles and terrain costs.
  - Dynamic maps: Includes patrolling vehicle (black triangle) that blocks paths, triggering replanning.
- **Dynamic Replanning**: Simulates agent movement; detects blocks and recomputes paths from the current position (e.g., 1-2 replans per run).
- **Visualization**: Matplotlib heatmaps showing start (green), goal (red), obstacles (dark), path (red line), and vehicle marker. Saves PNGs automatically.
- **Metrics**: Path cost (total distance/terrain), nodes expanded, execution time, and replan count.
- **Testing**: 6 unit tests for grid loading, planners, and edge cases (using pytest).
- **CLI Interface**: Easy command-line runs with flags for planner, map, replanning, and plotting.

## Quick Demo
Run A* on the small map to see an optimal path avoiding obstacles:


*(Red line: Heuristic-guided path from (0,0) to (4,4). Blue heatmap: Terrain costs; dark squares: obstacles.)*

## Setup
1. **Clone the Repository**:git clone https://github.com/prakhhhh/autonomous-delivery-agent.git cd autonomous-delivery-agent
2. **Install Dependencies** (Python 3.11+ required):- Core: NumPy (arrays/grids), Matplotlib (plots), pytest (tests).
3. **Run Tests** (Verify everything works):
4. pytest tests/ -v
5. python -m src.main --planner [bfs|ucs|astar|sa] --map [small|medium|large|dynamic] [--replan] [--plot]
6. python -m src.main --planner astar --map small --plot
7. python -m src.main --planner bfs --map small --plot python -m src.main --planner ucs --map small --plot python -m src.main --planner sa --map small --plot # Stochastic—rerun for variation
8. python -m src.main --planner astar --map large --plot
9. python -m src.main --planner astar --map dynamic --replan --plot python -m src.main --planner sa --map dynamic --replan --plot # Adapts better to uncertainty



Project Structure:
autonomous-delivery-agent/
├── .gitignore              # Ignores caches/logs
├── requirements.txt        # Dependencies
├── README.md              # This file
├── src/                   # Core code
│   ├── __init__.py
│   ├── main.py            # CLI entrypoint
│   ├── grid.py            # Map loading/parsing
│   └── planners.py        # BFS/UCS/A*/SA implementations
├── maps/                  # Input files
│   ├── small.map
│   ├── medium.map
│   ├── large.map
│   ├── dynamic.map
│   └── dynamic.dyn        # Vehicle patrol path
├── tests/                 # Unit tests
    ├── test_grid.py
    └── test_planners.py

Here's a demo video:
https://github.com/user-attachments/assets/18e30041-a321-4a13-9f2b-797ac9f8cdc4








