from __future__ import annotations

import argparse
import csv
import json
import sys
from glob import glob
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
    parser = argparse.ArgumentParser(description="Evaluate one checkpoint across multiple MuJoCo navigation maps.")
    parser.add_argument("--map-configs", nargs="+", required=True, help="Map YAML paths or glob patterns.")
    parser.add_argument("--checkpoint", default=None, help="Policy checkpoint. Omit only with --heuristic.")
    parser.add_argument("--run-name", required=True, help="Output folder under results/.")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--heuristic", action="store_true", help="Evaluate the built-in heuristic instead of a checkpoint.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--goal-observation-max", type=float, default=None)
    parser.add_argument("--auto-waypoints", action="store_true", help="Use A* waypoints during eval. Leave off for pure policy eval.")
    parser.add_argument("--waypoint-resolution", type=float, default=0.12)
    parser.add_argument("--waypoint-radius", type=float, default=0.45)
    parser.add_argument("--dense-waypoints", action="store_true")
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--no-png", action="store_true")
    return parser.parse_args()


def expand_map_configs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    resolved = []
    seen = set()
    for path in paths:
        resolved_path = path.resolve()
        if resolved_path in seen:
            continue
        if not resolved_path.exists():
            raise FileNotFoundError(f"Map config not found: {path}")
        seen.add(resolved_path)
        resolved.append(resolved_path)
    return resolved


def prepare_config(path: Path, args: argparse.Namespace) -> dict:
    cfg = load_task_config(path)
    if args.max_steps is not None:
        cfg["episode"]["max_steps"] = int(args.max_steps)
    if args.goal_observation_max is not None:
        cfg["goal"]["observation_max_distance"] = float(args.goal_observation_max)
    if args.auto_waypoints:
        cfg = with_auto_waypoints(
            cfg,
            resolution=args.waypoint_resolution,
            waypoint_radius=args.waypoint_radius,
            simplify=not args.dense_waypoints,
        )
    return cfg


def write_summary(summary: list[dict], output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "map",
        "episodes",
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "mean_steps",
        "mean_final_distance",
        "output_dir",
        "gif",
        "png",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "heuristic": bool(args.heuristic),
        "episodes": int(args.episodes),
        "num_envs": int(args.num_envs),
        "auto_waypoints": bool(args.auto_waypoints),
        "maps": summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.heuristic and not args.checkpoint:
        raise ValueError("--checkpoint is required unless --heuristic is used.")

    map_paths = expand_map_configs(args.map_configs)
    device = torch.device(args.device)
    model = None
    if not args.heuristic:
        model = load_policy_from_checkpoint(args.checkpoint, MujocoNavigationEnv.observation_dim, MujocoNavigationEnv.action_dim, device)

    output_dir = ROOT / "results" / args.run_name
    summary: list[dict] = []
    for map_index, map_path in enumerate(map_paths):
        cfg = prepare_config(map_path, args)
        map_name = str(cfg.get("name", map_path.stem))
        map_dir = output_dir / map_name
        metrics, episodes = evaluate_policy(
            cfg,
            model,
            episodes=args.episodes,
            num_envs=args.num_envs,
            device=device,
            seed=args.seed + map_index * 1000,
        )
        write_eval_outputs(metrics, episodes, map_dir / "eval.json", map_dir / "eval.csv")
        png_path = map_dir / "rollout.png"
        gif_path = map_dir / "rollout.gif"
        if not args.no_png:
            render_rollout_png(cfg, episodes[0], png_path)
        if not args.no_gif:
            render_rollout_gif(cfg, episodes[0], gif_path)
        row = {
            "map": map_name,
            "episodes": int(metrics["episodes"]),
            "success_rate": metrics["success_rate"],
            "collision_rate": metrics["collision_rate"],
            "timeout_rate": metrics["timeout_rate"],
            "mean_steps": metrics["mean_steps"],
            "mean_final_distance": metrics["mean_final_distance"],
            "output_dir": str(map_dir.resolve()),
            "gif": str(gif_path.resolve()) if not args.no_gif else "",
            "png": str(png_path.resolve()) if not args.no_png else "",
        }
        summary.append(row)
        print(
            f"{map_name}: success={row['success_rate']:.3f} "
            f"collision={row['collision_rate']:.3f} timeout={row['timeout_rate']:.3f} "
            f"steps={row['mean_steps']:.1f}"
        )

    write_summary(summary, output_dir, args)
    print(f"summary: {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
