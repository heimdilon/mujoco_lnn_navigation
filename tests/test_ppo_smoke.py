from pathlib import Path

from mujoco_lnn_nav.config import load_task_config, load_train_config
from mujoco_lnn_nav.envs.navigation import MujocoNavigationEnv
from mujoco_lnn_nav.models.policies import build_actor_critic
from mujoco_lnn_nav.training.ppo import PPOConfig, train_ppo


def test_ppo_one_update_smoke():
    task_cfg = load_task_config("configs/task/sparse_goal.yaml")
    train_cfg = load_train_config("configs/train/ppo_mlp.yaml")
    train_cfg["total_steps"] = 64
    train_cfg["num_envs"] = 2
    train_cfg["rollout_steps"] = 16
    train_cfg["minibatch_size"] = 16
    train_cfg["update_epochs"] = 1
    env = MujocoNavigationEnv(task_cfg, num_envs=2, seed=321)
    model = build_actor_critic("mlp", env.observation_dim, env.action_dim, 32)
    run_dir = Path("results/test_ppo_smoke")
    history = train_ppo(env, model, PPOConfig.from_dict(train_cfg), "mlp", run_dir, train_cfg)
    assert history
    assert (run_dir / "latest.pt").exists()
