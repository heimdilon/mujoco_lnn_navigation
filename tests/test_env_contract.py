import numpy as np
import torch

from mujoco_lnn_nav.config import load_task_config
from mujoco_lnn_nav.envs.navigation import MujocoNavigationEnv


def test_reset_step_shapes():
    cfg = load_task_config("configs/task/sparse_goal.yaml")
    env = MujocoNavigationEnv(cfg, num_envs=3, seed=123)
    obs = env.reset()
    assert obs.shape == (3, 38)
    out = env.step(torch.zeros((3, 2)))
    assert out.observation.shape == (3, 38)
    assert out.reward.shape == (3,)
    assert out.terminated.shape == (3,)
    assert out.truncated.shape == (3,)


def test_success_reward_flag():
    cfg = load_task_config("configs/task/sparse_goal.yaml")
    env = MujocoNavigationEnv(cfg, num_envs=1, seed=123, auto_reset=False)
    single = env.envs[0]
    single.goal = single.position + np.array([0.03, 0.0], dtype=np.float32)
    single.prev_distance = single.goal_distance()
    out = env.step(torch.zeros((1, 2)))
    assert bool(out.info["success"][0])
    assert bool(out.terminated[0])
    assert float(out.reward[0]) > 1.0


def test_timeout_flag():
    cfg = load_task_config("configs/task/sparse_goal.yaml")
    cfg["episode"]["max_steps"] = 1
    env = MujocoNavigationEnv(cfg, num_envs=1, seed=123, auto_reset=False)
    out = env.step(torch.zeros((1, 2)))
    assert bool(out.truncated[0]) or bool(out.terminated[0])

