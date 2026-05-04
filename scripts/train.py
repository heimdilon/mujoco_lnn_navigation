from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mujoco_lnn_nav.config import load_task_config, load_train_config
from mujoco_lnn_nav.envs.navigation import MujocoNavigationEnv
from mujoco_lnn_nav.models.policies import build_actor_critic
from mujoco_lnn_nav.training.ppo import PPOConfig, train_ppo
from mujoco_lnn_nav.utils.evaluation import evaluate_policy, write_eval_outputs
from mujoco_lnn_nav.utils.rendering import render_rollout_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--resume", default=None, help="Resume model and optimizer state from a checkpoint.")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_cfg = load_task_config(args.task_config)
    train_cfg = load_train_config(args.train_config)
    if args.steps is not None:
        train_cfg["total_steps"] = args.steps
    if args.num_envs is not None:
        train_cfg["num_envs"] = args.num_envs
    if args.eval_episodes is not None:
        train_cfg["eval_episodes"] = args.eval_episodes
    if args.eval_interval is not None:
        train_cfg["eval_interval"] = args.eval_interval

    device = torch.device(args.device)
    ppo_cfg = PPOConfig.from_dict(train_cfg)
    env = MujocoNavigationEnv(task_cfg, num_envs=ppo_cfg.num_envs, device=device, auto_reset=True)
    policy_name = str(train_cfg.get("policy", "mlp"))
    model = build_actor_critic(policy_name, env.observation_dim, env.action_dim, int(train_cfg.get("hidden_size", 128))).to(device)
    start_step = 0
    optimizer_state = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        if checkpoint.get("policy", policy_name) != policy_name:
            raise ValueError(f"Checkpoint policy {checkpoint.get('policy')} does not match config policy {policy_name}.")
        model.load_state_dict(checkpoint["model_state"])
        optimizer_state = checkpoint.get("optimizer_state")
        start_step = int(checkpoint.get("step", 0))
    run_dir = ROOT / "results" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    def eval_fn(policy, step: int) -> dict[str, float]:
        metrics, episodes = evaluate_policy(
            task_cfg,
            policy,
            episodes=int(train_cfg.get("eval_episodes", 32)),
            num_envs=1,
            device=device,
            seed=5000 + step,
            deterministic=True,
        )
        write_eval_outputs(metrics, episodes, run_dir / f"eval_{step}.json", run_dir / f"eval_{step}.csv")
        render_rollout_png(task_cfg, episodes[0], run_dir / f"rollout_{step}.png")
        return metrics

    history = train_ppo(
        env,
        model,
        ppo_cfg,
        policy_name,
        run_dir,
        train_cfg,
        eval_fn=eval_fn,
        eval_interval=int(train_cfg.get("eval_interval", 10000)),
        start_step=start_step,
        optimizer_state=optimizer_state,
    )
    with (run_dir / "train_history.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in history for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    metrics, episodes = evaluate_policy(task_cfg, model, episodes=int(train_cfg.get("eval_episodes", 32)), num_envs=1, device=device)
    write_eval_outputs(metrics, episodes, run_dir / "eval_latest.json", run_dir / "eval_latest.csv")
    render_rollout_png(task_cfg, episodes[0], run_dir / "rollout_latest.png")
    print(metrics)


if __name__ == "__main__":
    main()
