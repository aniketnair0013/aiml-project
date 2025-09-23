from src.grid import GridEnvironment
from src.planners import Planner
import logging

class DeliveryAgent:
    def __init__(self, env: GridEnvironment, planner_type: str):
        self.env = env
        self.planner = Planner(env)
        self.planner_type = planner_type
        self.max_fuel = 1000
        self.max_steps = 200
        logging.basicConfig(level=logging.INFO, filename='replan_log.txt', filemode='w')

    def plan_path(self, start: Tuple[int, int]) -> List[Tuple[int, int]]:
        if self.planner_type == 'bfs':
            return self.planner.bfs(start)[0]
        elif self.planner_type == 'ucs':
            return self.planner.ucs(start)[0]
        elif self.planner_type == 'astar':
            return self.planner.astar(start)[0]
        elif self.planner_type == 'sa':
            return self.planner.simulated_annealing(start, self.max_steps)[0]
        return []

    def execute_with_replanning(self, path: List[Tuple[int, int]], enable_replan: bool = False):
        current_pos = path[0]
        step = 0
        while step < len(path) and self._check_constraints(step):
            if enable_replan and self.env.is_occupied(path[step], step):
                logging.info(f"Step {step}: Obstacle at {path[step]}, replanning...")
                new_path = self.plan_path(current_pos)
                if new_path:
                    path = new_path  # Replan from current
                    logging.info(f"Replanned path length: {len(path)}")
                else:
                    break
            current_pos = path[step]
            step += 1
            print(f"Step {step}: At {current_pos}")
        if current_pos == self.env.goal:
            print("Delivery successful!")
        else:
            print("Failed: Constraints violated or blocked.")

    def _check_constraints(self, steps: int) -> bool:
        return steps < self.max_steps  # Fuel checked in path cost