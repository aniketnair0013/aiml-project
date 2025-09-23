# Autonomous Delivery Agent Project Report (CSA2001)

## Environment Model
A 2D grid with rows and columns up to 20x20 makes up the environment.  Static obstacles cost -1; cells cost ≥1 (terrain).  Vehicles and other dynamic obstacles follow known cyclic paths (horizon=10 steps) in a deterministic manner.  The agent delivers to (gx,gy) after starting at (sx,sy).  Condition: (x,y,time).  Actions: no diagonals, four connected moves.  Cost is the sum of the terrain and the dynamics' implicit time.  Limitations: ≤1000 for the fuel and ≤200 steps for the time.

## Design of Agents
The agent chooses routes that minimize the overall cost because it is logical.  Organizers:
BFS: Ignorant, considers every edge to be the same (steps as cost).
Uninformed, UCS takes terrain costs into consideration.
- A*: Well-informed, employs the admissible Manhattan heuristic \$h(n) = |x_g - x_n| + |y_g - y_n|\$ (ignoring costs/obstacles, ≤ true cost since min cost=1).
SA: Local search for replanning—begins with a greedy path, perturbs (swaps/inserts), and accepts worse with \$P = e^{-\Delta / T}\$ (T cools from 1000).  uses collision-based replanning to handle dynamics.

Replanning: Use SA (fast for local changes) to replan from the current position if a dynamic obstacle blocks the next cell at time t during execution.  Proof of concept:  The log indicates an obstacle at t=5, and the replan lowers the cost by 2.

## Test Findings
Four maps were used for testing: small (5x5), medium (10x10), large (20x20), and dynamic (10x10 w/1 moving obs).  ran each planner three times (average metrics).  Hardware: A typical laptop.

*Table 1: Metrics (Time in seconds, Nodes Expanded, Average Cost)*
