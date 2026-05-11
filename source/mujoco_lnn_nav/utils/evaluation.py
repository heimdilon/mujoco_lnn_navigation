from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from mujoco_lnn_nav.envs.navigation import MujocoNavigationEnv
from mujoco_lnn_nav.envs.rays import wrap_angle
from mujoco_lnn_nav.models.policies import BaseActorCritic


def heuristic_action(env: MujocoNavigationEnv) -> torch.Tensor:
    actions = []
    for single in env.envs:
        goal_vec = single.goal - single.position
        bearing = float(wrap_angle(np.arctan2(goal_vec[1], goal_vec[0]) - single.yaw))
        rays = single.observation()[6:]
        front = float(np.min(np.concatenate([rays[:3], rays[-3:]])))
        left = float(np.mean(rays[4:12]))
        right = float(np.mean(rays[-12:-4]))
        angular = np.clip(2.8 * bearing / np.pi, -1.0, 1.0)
        linear = np.clip(np.cos(bearing), 0.0, 1.0)
        if abs(bearing) > 0.9:
            linear *= 0.25
        if front < 0.28:
            linear *= 0.15
            angular = np.clip(angular + (0.8 if right > left else -0.8), -1.0, 1.0)
        actions.append([linear, angular])
    return torch.as_tensor(actions, dtype=torch.float32, device=env.device)


def evaluate_policy(
    config: dict,
    model: BaseActorCritic | None,
    episodes: int = 32,
    num_envs: int = 1,
    device: str | torch.device = "cpu",
    seed: int = 1000,
    deterministic: bool = True,
    action_fn: Callable[[MujocoNavigationEnv], torch.Tensor] | None = None,
) -> tuple[dict[str, float], list[dict]]:
    env = MujocoNavigationEnv(config, num_envs=num_envs, device=device, seed=seed, auto_reset=False)
    completed: list[dict] = []
    obs = env.reset()
    active_steps = np.zeros((num_envs,), dtype=np.int32)
    recurrent_state = None
    if model is not None and hasattr(model, "initial_state"):
        recurrent_state = model.initial_state(num_envs, env.device)

    while len(completed) < episodes:
        if action_fn is not None:
            action = action_fn(env)
        elif model is None:
            action = heuristic_action(env)
        elif recurrent_state is not None and hasattr(model, "act_recurrent"):
            with torch.no_grad():
                action, _, _, recurrent_state = model.act_recurrent(obs, recurrent_state, deterministic=deterministic)
        else:
            with torch.no_grad():
                action, _, _ = model.act(obs, deterministic=deterministic)
        out = env.step(action)
        active_steps += 1
        obs = out.observation
        done = out.done.detach().cpu().numpy()
        for idx, is_done in enumerate(done):
            if not is_done:
                continue
            raw = out.info["raw"][idx]
            completed.append(
                {
                    "episode": len(completed),
                    "success": bool(raw["success"]),
                    "collision": bool(raw["collision"]),
                    "timeout": bool(raw["timeout"]),
                    "steps": int(active_steps[idx]),
                    "final_distance": float(raw["distance"]),
                    "path": list(env.envs[idx].last_path),
                    "yaws": list(env.envs[idx].last_yaws),
                    "distances": list(env.envs[idx].last_distances),
                    "obstacles": [obs_spec.__dict__ for obs_spec in env.envs[idx].obstacles],
                    "obstacle_paths": list(env.envs[idx].last_obstacle_paths),
                    "goal": getattr(env.envs[idx], "final_goal", env.envs[idx].goal).astype(float).tolist(),
                    "waypoints": [point.astype(float).tolist() for point in getattr(env.envs[idx], "waypoints", [])],
                }
            )
            env.envs[idx].reset()
            if recurrent_state is not None and hasattr(model, "reset_state_indices"):
                recurrent_state = model.reset_state_indices(recurrent_state, [idx])
            active_steps[idx] = 0
            if len(completed) >= episodes:
                break
        obs = torch.as_tensor(np.stack([single.observation() for single in env.envs]), dtype=torch.float32, device=env.device)

    successes = sum(1 for ep in completed if ep["success"])
    collisions = sum(1 for ep in completed if ep["collision"])
    timeouts = sum(1 for ep in completed if ep["timeout"])
    metrics = {
        "episodes": float(len(completed)),
        "success_rate": successes / max(1, len(completed)),
        "collision_rate": collisions / max(1, len(completed)),
        "timeout_rate": timeouts / max(1, len(completed)),
        "mean_steps": float(np.mean([ep["steps"] for ep in completed])),
        "mean_final_distance": float(np.mean([ep["final_distance"] for ep in completed])),
    }
    return metrics, completed


def write_eval_outputs(metrics: dict[str, float], episodes: list[dict], output_json: Path, output_csv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"metrics": metrics, "episodes": episodes}, indent=2), encoding="utf-8")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode", "success", "collision", "timeout", "steps", "final_distance"])
        writer.writeheader()
        for ep in episodes:
            writer.writerow({key: ep[key] for key in writer.fieldnames})
