import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from grid import GridEnvironment
from planners import Planner

@pytest.fixture
def env():
    return GridEnvironment('maps/small.map')

@pytest.fixture
def planner(env):
    return Planner(env)

def test_bfs(planner, env):
    path, metrics = planner.bfs(env.start)
    assert len(path) > 0
    assert path[-1] == env.goal
    assert metrics['cost'] < float('inf')

def test_astar(planner, env):
    path, metrics = planner.astar(env.start)
    assert len(path) > 0
    assert path[-1] == env.goal
    assert metrics['nodes'] < 25  # Small grid limit

def test_sa(planner, env):
    path, metrics = planner.simulated_annealing(env.start)
    assert len(path) > 0 or metrics['cost'] < float('inf')