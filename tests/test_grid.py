import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from grid import GridEnvironment

@pytest.fixture
def small_env():
    env = GridEnvironment('maps/small.map')
    assert env.grid.shape == (5, 5)  # Early fail if load wrong
    return env

def test_load_map(small_env):
    assert small_env.start == (0, 0)
    assert small_env.goal == (4, 4)
    assert small_env.get_cost((0, 0)) == 1
    assert small_env.get_cost((1, 1)) == float('inf')  # Obstacle at (1,1)

def test_successors(small_env):
    succ = small_env.successors((0, 0), 0)
    assert len(succ) >= 2  # At least down and right
    assert any(pos == (0, 1) for pos, _ in succ)  # Right
    assert any(pos == (1, 0) for pos, _ in succ)  # Down

def test_dynamic_occupied():
    dyn_file = 'maps/dynamic.dyn'
    dyn_env = GridEnvironment('maps/dynamic.map', dyn_file if os.path.exists(dyn_file) else None)
    # Basic check: Not occupied initially at start
    assert dyn_env.is_occupied((0, 0), 0) == False  # Start free
    if os.path.exists(dyn_file):
        # Check actual path: Occupied at (5,6) at t=0, (5,7) at t=1 (based on your .dyn)
        assert dyn_env.is_occupied((5, 6), 0) == True
        assert dyn_env.is_occupied((5, 7), 1) == True
        # Cycle check
        assert dyn_env.is_occupied((5, 6), 4) == True  # 4 % len(path) = 0
    else:
        # Fallback: No dynamics, so False for dynamic positions
        assert dyn_env.is_occupied((5, 6), 1) == False
