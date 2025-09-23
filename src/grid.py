import numpy as np
import os
from typing import List, Tuple, Dict

class GridEnvironment:
    def __init__(self, map_file: str, dyn_file: str = None):
        if not os.path.exists(map_file):
            raise FileNotFoundError(f"Map file not found: {map_file}. Ensure maps/ directory has the file.")
        self.grid = self._load_map(map_file)
        self.rows, self.cols = self.grid.shape
        # start and goal are now set in _load_map; no need for _parse_positions
        self.obstacles = self._find_static_obstacles()
        self.moving_obstacles: Dict[str, Dict] = self._load_dynamic(dyn_file) if dyn_file else {}
        self.horizon = 10  # Planning lookahead for dynamics

    def _load_map(self, file: str) -> np.ndarray:
        try:
            with open(file, 'r') as f:
                lines = f.readlines()
                if len(lines) < 2:
                    raise ValueError(f"Map file {file} is empty or malformed. Needs at least 2 lines: positions + grid.")
                pos_line = lines[0].strip().split()
                if len(pos_line) != 4:
                    raise ValueError(f"First line of {file} must have 4 integers: start_x start_y goal_x goal_y")
                self.start = (int(pos_line[0]), int(pos_line[1]))
                self.goal = (int(pos_line[2]), int(pos_line[3]))
                grid_data = []
                for line in lines[1:]:
                    row = [int(x) for x in line.strip().split() if x.strip()]  # Ignore empty
                    if len(row) == 0:
                        continue
                    grid_data.append(row)
                if not grid_data:
                    raise ValueError(f"No grid data in {file} after first line.")
                grid = np.array(grid_data, dtype=int)
                # Ensure rectangular (all rows same length)
                if not np.all(np.array([len(row) for row in grid_data]) == grid.shape[1]):
                    raise ValueError(f"Grid in {file} must be rectangular (equal row lengths).")
                grid[grid == 0] = 1  # Normalize free to cost 1
                # Replace any non-obstacle negatives with -1 if needed, but assume format is correct
                return grid
        except (ValueError, IndexError, TypeError) as e:
            raise ValueError(f"Error parsing map {file}: {e}. Check format: First line 'sx sy gx gy', then grid rows.")

    def _find_static_obstacles(self) -> List[Tuple[int, int]]:
        return [(i, j) for i in range(self.rows) for j in range(self.cols) if self.grid[i, j] == -1]

    def _load_dynamic(self, file: str) -> Dict[str, Dict]:
        obs = {}
        if not file or not os.path.exists(file):
            return obs  # Graceful fallback
        try:
            with open(file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if len(parts) < 4:
                        print(f"Warning: Skipping invalid line {line_num} in {file}: {line.strip()}")
                        continue
                    obs_id = parts[0]
                    try:
                        sx, sy = int(parts[1]), int(parts[2])
                        path_len = int(parts[3])  # Number of coordinates (even)
                        if path_len % 2 != 0:
                            raise ValueError("path_len must be even (x y pairs)")
                        path = []
                        for i in range(path_len // 2):
                            px = int(parts[4 + 2*i])
                            py = int(parts[4 + 2*i + 1])
                            path.append((px, py))
                        obs[obs_id] = {'pos': (sx, sy), 'path': path, 'speed': 1}
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Skipping obstacle {obs_id} in {file}: {e}")
            return obs
        except Exception as e:
            print(f"Warning: Failed to load dynamic file {file}: {e}. Using no dynamics.")
            return {}

    def get_cost(self, pos: Tuple[int, int]) -> int:
        x, y = pos
        if 0 <= x < self.rows and 0 <= y < self.cols and self.grid[x, y] != -1:
            return self.grid[x, y]
        return float('inf')

    def is_occupied(self, pos: Tuple[int, int], time: int) -> bool:
        # Check static
        if self.get_cost(pos) == float('inf'):
            return True
        # Check dynamic at time t (mod path length)
        for obs in self.moving_obstacles.values():
            path = obs['path']
            path_len = len(path)
            if path_len > 0:
                obs_pos = path[time % path_len]
                if obs_pos == pos:
                    return True
        return False

    def successors(self, pos: Tuple[int, int], time: int) -> List[Tuple[Tuple[int, int], int]]:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-connected
        succ = []
        for dx, dy in dirs:
            new_pos = (pos[0] + dx, pos[1] + dy)
            if not self.is_occupied(new_pos, time + 1):
                cost = self.get_cost(new_pos)
                if cost != float('inf'):
                    succ.append((new_pos, cost))
        return succ 