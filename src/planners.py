import heapq
from typing import List, Tuple, Dict
from grid import GridEnvironment
import time
import random
import numpy as np


# import time
# import heapq
# import random
# import math
# import numpy as np
# from grid import GridEnvironment  # Absolute import (sys.path will handle location)
# # ... (rest of your planners.py code: class Planner, methods bfs/ucs/etc.)

class Planner:
    def __init__(self, env: GridEnvironment):
        self.env = env

    def manhattan_heuristic(self, pos: Tuple[int, int]) -> int:
        return abs(pos[0] - self.env.goal[0]) + abs(pos[1] - self.env.goal[1])

    def bfs(self, start: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], Dict]:
        start_time = time.time()
        queue = [(start, [start])]  # (pos, path)
        visited = set([start])
        nodes_expanded = 0
        while queue:
            pos, path = queue.pop(0)
            nodes_expanded += 1
            if pos == self.env.goal:
                return path, {'cost': len(path) - 1, 'nodes': nodes_expanded, 'time': time.time() - start_time}
            for next_pos, _ in self.env.successors(pos, len(path)):
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [next_pos]))
        return [], {'cost': float('inf'), 'nodes': nodes_expanded, 'time': time.time() - start_time}

    def ucs(self, start: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], Dict]:
        start_time = time.time()
        pq = [(0, start, [start])]  # (cost, pos, path)
        visited = {start: 0}
        nodes_expanded = 0
        while pq:
            cost, pos, path = heapq.heappop(pq)
            nodes_expanded += 1
            if pos == self.env.goal:
                return path, {'cost': cost, 'nodes': nodes_expanded, 'time': time.time() - start_time}
            for next_pos, edge_cost in self.env.successors(pos, len(path)):
                new_cost = cost + edge_cost
                if new_cost < visited.get(next_pos, float('inf')):
                    visited[next_pos] = new_cost
                    heapq.heappush(pq, (new_cost, next_pos, path + [next_pos]))
        return [], {'cost': float('inf'), 'nodes': nodes_expanded, 'time': time.time() - start_time}

    def astar(self, start: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], Dict]:
        start_time = time.time()
        pq = [(0 + self.manhattan_heuristic(start), 0, start, [start])]  # (f, g, pos, path)
        visited = {start: 0}
        nodes_expanded = 0
        while pq:
            _, g, pos, path = heapq.heappop(pq)
            nodes_expanded += 1
            if pos == self.env.goal:
                return path, {'cost': g, 'nodes': nodes_expanded, 'time': time.time() - start_time}
            for next_pos, edge_cost in self.env.successors(pos, len(path)):
                new_g = g + edge_cost
                if new_g < visited.get(next_pos, float('inf')):
                    h = self.manhattan_heuristic(next_pos)
                    visited[next_pos] = new_g
                    heapq.heappush(pq, (new_g + h, new_g, next_pos, path + [next_pos]))
        return [], {'cost': float('inf'), 'nodes': nodes_expanded, 'time': time.time() - start_time}

    def simulated_annealing(self, start: Tuple[int, int], max_steps: int = 200) -> Tuple[List[Tuple[int, int]], Dict]:
        # Local search for replanning: Start with greedy path, perturb and anneal
        random.seed(42)
        current_path = self._greedy_path(start)  # Initial hill-climb approximation
        if not current_path:
            return [], {'cost': float('inf'), 'nodes': 0, 'time': 0}
        current_cost = self._path_cost(current_path)
        best_path, best_cost = current_path[:], current_cost
        temp = 1000
        cooling = 0.995
        nodes_expanded = len(current_path)  # Approximate
        start_time = time.time()

        for _ in range(max_steps):
            # Perturb: Random swap or insert wait
            new_path = self._perturb_path(current_path)
            new_cost = self._path_cost(new_path)
            delta = new_cost - current_cost
            if delta < 0 or random.random() < np.exp(-delta / temp):
                current_path, current_cost = new_path, new_cost
                if current_cost < best_cost:
                    best_path, best_cost = current_path[:], current_cost
            temp *= cooling
            nodes_expanded += 1  # Per iteration

        return best_path, {'cost': best_cost, 'nodes': nodes_expanded, 'time': time.time() - start_time}

    def _greedy_path(self, start: Tuple[int, int]) -> List[Tuple[int, int]]:
        path = [start]
        pos = start
        while pos != self.env.goal:
            succ = self.env.successors(pos, len(path))
            if not succ:
                return []
            # Greedy: Min heuristic successor
            next_pos = min(succ, key=lambda x: self.manhattan_heuristic(x[0]))
            path.append(next_pos[0])
            pos = next_pos[0]
        return path

    def _path_cost(self, path: List[Tuple[int, int]]) -> int:
        cost = 0
        for i in range(len(path) - 1):
            edge_cost = self.env.get_cost(path[i+1])
            if edge_cost == float('inf'):
                return float('inf')
            cost += edge_cost
        return cost

    def _perturb_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        # Random restart-like: Swap two points or add detour
        idx1, idx2 = random.sample(range(1, len(path)-1), 2)
        new_path = path[:]
        new_path[idx1], new_path[idx2] = new_path[idx2], new_path[idx1]
        # Ensure validity (simple check)
        if self._path_cost(new_path) == float('inf'):
            return path  # Revert
        return new_path