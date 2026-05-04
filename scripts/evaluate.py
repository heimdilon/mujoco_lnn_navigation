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
from mujoco_lnn_nav.utils.evaluation import evaluate_policy, write_eval_outputs
from mujoco_lnn_nav.utils.planning import with_auto_waypoints
from mujoco_lnn_nav.utils.rendering import render_rollout_gif, render_rollout_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--run-name", default="eval")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--heuristic", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--goal-observation-max", type=float, default=None)
    parser.add_argument("--auto-waypoints", action="store_true")
    parser.add_argument("--waypoint-resolution", type=float, default=0.12)
    parser.add_argument("--waypoint-radius", type=float, default=0.45)
    parser.add_argument("--dense-waypoints", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_cfg = load_task_config(args.task_config)
    if args.max_steps is not None:
        task_cfg["episode"]["max_steps"] = int(args.max_steps)
    if args.goal_observation_max is not None:
        task_cfg["goal"]["observation_max_distance"] = float(args.goal_observation_max)
    if args.auto_waypoints:
        task_cfg = with_auto_waypoints(
            task_cfg,
            resolution=args.waypoint_resolution,
            waypoint_radius=args.waypoint_radius,
            simplify=not args.dense_waypoints,
        )
    device = torch.device(args.device)
    model = None
    if args.checkpoint and not args.heuristic:
        model = load_policy_from_checkpoint(args.checkpoint, MujocoNavigationEnv.observation_dim, MujocoNavigationEnv.action_dim, device)
    metrics, episodes = evaluate_policy(task_cfg, model, episodes=args.episodes, num_envs=args.num_envs, device=device)
    out_dir = ROOT / "results" / args.run_name
    write_eval_outputs(metrics, episodes, out_dir / "eval.json", out_dir / "eval.csv")
    render_rollout_png(task_cfg, episodes[0], out_dir / "rollout.png")
    render_rollout_gif(task_cfg, episodes[0], out_dir / "rollout.gif")
    print(metrics)


if __name__ == "__main__":
    main()
