from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mujoco_lnn_nav.config import load_task_config
from mujoco_lnn_nav.envs.navigation import MujocoNavigationEnv
from mujoco_lnn_nav.utils.checkpoints import load_policy_from_checkpoint
from mujoco_lnn_nav.utils.evaluation import evaluate_policy
from mujoco_lnn_nav.utils.rendering import render_rollout_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--run-name", default="diagnose")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_cfg = load_task_config(args.task_config)
    device = torch.device(args.device)
    model = None
    if args.checkpoint:
        model = load_policy_from_checkpoint(args.checkpoint, MujocoNavigationEnv.observation_dim, MujocoNavigationEnv.action_dim, device)
    metrics, episodes = evaluate_policy(task_cfg, model, episodes=args.episodes, num_envs=1, device=device)
    out_dir = ROOT / "results" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, episode in enumerate(episodes):
        render_rollout_png(task_cfg, episode, out_dir / f"rollout_{idx:03d}.png")
    print(metrics)
    for episode in episodes:
        distances = episode["distances"]
        if len(distances) >= 2:
            print(
                f"episode={episode['episode']} success={episode['success']} "
                f"distance={distances[0]:.3f}->{distances[-1]:.3f} steps={episode['steps']}"
            )


if __name__ == "__main__":
    main()

