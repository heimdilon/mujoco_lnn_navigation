import numpy as np

from mujoco_lnn_nav.config import load_task_config
from mujoco_lnn_nav.envs.layouts import fixed_obstacles, has_fixed_map, is_free, max_obstacles
from mujoco_lnn_nav.envs.navigation import MujocoNavigationEnv


def test_fixed_map_reset_uses_manual_start_goal_obstacles():
    cfg = load_task_config("configs/task/sparse_goal.yaml")
    cfg["map"] = {
        "enabled": True,
        "name": "unit_map",
        "start": [-1.0, -0.5, 0.25],
        "goal": [1.25, 0.75],
        "obstacles": [
            {"id": "c0", "shape": "cylinder", "x": 0.0, "y": 0.0, "radius": 0.2},
            {"id": "b0", "shape": "box", "x": 0.6, "y": -0.5, "half_x": 0.15, "half_y": 0.25, "yaw": 0.4},
        ],
    }
    assert has_fixed_map(cfg)
    assert max_obstacles(cfg) >= 2
    assert len(fixed_obstacles(cfg)) == 2
    env = MujocoNavigationEnv(cfg, num_envs=1, seed=1, auto_reset=False)
    obs = env.reset()
    single = env.envs[0]
    assert obs.shape == (1, 38)
    assert np.allclose(single.position, [-1.0, -0.5], atol=1e-4)
    assert np.allclose(single.goal, [1.25, 0.75], atol=1e-4)
    assert len(single.obstacles) == 2
    assert abs(single.obstacles[1].yaw - 0.4) < 1e-6


def test_rotated_wall_blocks_points_along_its_axis():
    cfg = load_task_config("configs/task/sparse_goal.yaml")
    cfg["map"] = {
        "enabled": True,
        "name": "wall_map",
        "start": [-1.0, -1.0, 0.0],
        "goal": [1.0, 1.0],
        "obstacles": [
            {"id": "wall", "shape": "box", "kind": "wall", "x": 0.0, "y": 0.0, "half_x": 0.7, "half_y": 0.08, "yaw": 0.785398},
        ],
    }
    wall = fixed_obstacles(cfg)[0]
    assert not is_free(np.array([0.25, 0.25], dtype=np.float32), [wall], cfg)
    assert is_free(np.array([0.25, -0.25], dtype=np.float32), [wall], cfg)


def test_dynamic_obstacle_motion_updates_obstacle_state_and_history():
    cfg = load_task_config("configs/task/sparse_goal.yaml")
    cfg["map"] = {
        "enabled": True,
        "name": "dynamic_unit_map",
        "start": [-1.5, -1.5, 0.0],
        "goal": [1.5, 1.5],
        "obstacles": [],
        "dynamic_obstacles": [
            {
                "id": "moving",
                "shape": "cylinder",
                "x": 0.0,
                "y": 0.0,
                "radius": 0.2,
                "motion": {"type": "line", "axis": [1.0, 0.0], "amplitude": 0.5, "period": 4.0},
            }
        ],
    }
    env = MujocoNavigationEnv(cfg, num_envs=1, seed=1, auto_reset=False)
    env.reset()
    single = env.envs[0]
    assert np.allclose(single.obstacles[0].center, [0.0, 0.0], atol=1e-6)

    env.step(np.zeros((1, 2), dtype=np.float32))

    expected_x = 0.5 * np.sin(2.0 * np.pi * single.time / 4.0)
    assert np.isclose(single.obstacles[0].x, expected_x, atol=1e-6)
    assert len(single.last_obstacle_paths[0]) == 2
