from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mujoco_lnn_nav.config import load_task_config, load_train_config, load_yaml
from mujoco_lnn_nav.envs.navigation import MujocoNavigationEnv
from mujoco_lnn_nav.models.policies import RecurrentActorCritic, build_actor_critic
from mujoco_lnn_nav.utils.evaluation import evaluate_policy, write_eval_outputs
from mujoco_lnn_nav.utils.planning import with_auto_waypoints
from mujoco_lnn_nav.utils.rendering import render_rollout_gif, render_rollout_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-configs", nargs="*", default=None)
    parser.add_argument("--split-config", default=None, help="Optional split YAML; train_maps are used for training.")
    parser.add_argument("--include-holdout-maps", action="store_true", help="Also train on holdout_maps from --split-config.")
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--policy", choices=["mlp", "cfc", "lnn", "gru", "lstm"], default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=20)
    parser.add_argument("--dagger-iterations", type=int, default=0)
    parser.add_argument("--dagger-rollouts-per-map", type=int, default=2)
    parser.add_argument("--dagger-epochs", type=int, default=40)
    parser.add_argument("--dagger-noise", type=float, default=0.06)
    parser.add_argument("--dagger-expert-mix", type=float, default=0.20)
    parser.add_argument("--no-final-eval", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def resolve_map_configs(args: argparse.Namespace) -> list[str]:
    map_paths: list[str] = []
    if args.split_config:
        split_cfg = load_yaml(args.split_config)
        map_paths.extend(str(path) for path in split_cfg.get("train_maps", []))
        if args.include_holdout_maps:
            map_paths.extend(str(path) for path in split_cfg.get("holdout_maps", []))
    if args.map_configs:
        map_paths.extend(str(path) for path in args.map_configs)
    if not map_paths:
        raise ValueError("Provide --map-configs or --split-config.")
    return map_paths


def teacher_action_to_goal(single, goal: np.ndarray) -> np.ndarray:
    vec = goal - single.position
    distance = float(np.linalg.norm(vec))
    bearing = float((np.arctan2(vec[1], vec[0]) - single.yaw + np.pi) % (2.0 * np.pi) - np.pi)
    angular = float(np.clip(5.0 * bearing / np.pi, -1.0, 1.0))
    linear = float(np.clip(0.95 * np.cos(bearing), 0.0, 0.95))
    if abs(bearing) > 0.85:
        linear *= 0.15
    elif abs(bearing) > 0.50:
        linear *= 0.35
    if distance < 0.34:
        linear = min(linear, 0.30)
    rays = single.observation()[6:]
    front = float(np.min(np.concatenate([rays[:2], rays[-2:]])))
    if front < 0.04:
        linear = 0.0
    return np.array([linear, angular], dtype=np.float32)


def teacher_action(single) -> np.ndarray:
    return teacher_action_to_goal(single, single.goal)


def prepare_teacher_config(path: str, train_cfg: dict) -> dict:
    map_cfg = load_task_config(path)
    map_cfg["episode"]["max_steps"] = int(train_cfg.get("max_steps", 900))
    map_cfg["goal"]["observation_max_distance"] = float(train_cfg.get("goal_observation_max", 10.0))
    map_cfg = with_auto_waypoints(
        map_cfg,
        resolution=float(train_cfg.get("waypoint_resolution", 0.10)),
        waypoint_radius=float(train_cfg.get("waypoint_radius", 0.30)),
        simplify=not bool(train_cfg.get("dense_waypoints", True)),
    )
    map_cfg["reward"]["waypoint_bonus"] = 1.0
    return map_cfg


def collect_sequences(map_paths: list[str], train_cfg: dict, device: torch.device) -> tuple[list[torch.Tensor], list[torch.Tensor], list[dict]]:
    obs_sequences: list[torch.Tensor] = []
    action_sequences: list[torch.Tensor] = []
    metadata: list[dict] = []
    for map_path in map_paths:
        teacher_cfg = prepare_teacher_config(map_path, train_cfg)
        for episode_idx in range(int(train_cfg.get("episodes_per_map", 4))):
            env = MujocoNavigationEnv(teacher_cfg, num_envs=1, device=device, seed=100 + episode_idx, auto_reset=False)
            env.reset()
            obs_list: list[np.ndarray] = []
            action_list: list[np.ndarray] = []
            final_info = None
            for _ in range(int(train_cfg.get("max_steps", 900))):
                single = env.envs[0]
                obs_list.append(single.observation_for_goal(single.final_goal))
                action = teacher_action(single)
                action_list.append(action)
                out = env.step(torch.as_tensor(action, dtype=torch.float32, device=device).view(1, 2))
                final_info = out.info["raw"][0]
                if bool(out.done[0]):
                    break
            if not obs_list:
                continue
            success = bool(final_info and final_info["success"])
            metadata.append(
                {
                    "map": teacher_cfg["name"],
                    "steps": len(obs_list),
                    "success": success,
                    "collision": bool(final_info and final_info["collision"]),
                    "timeout": bool(final_info and final_info["timeout"]),
                }
            )
            if not success:
                continue
            obs_sequences.append(torch.as_tensor(np.stack(obs_list), dtype=torch.float32, device=device))
            action_sequences.append(torch.as_tensor(np.stack(action_list), dtype=torch.float32, device=device))
    if not obs_sequences:
        raise RuntimeError("No successful teacher sequences were collected; cannot run behavioral cloning.")
    return obs_sequences, action_sequences, metadata


def waypoint_index_for_position(waypoints: list[np.ndarray], position: np.ndarray, radius: float, current: int = 0) -> int:
    index = current
    while index < len(waypoints) - 1 and np.linalg.norm(waypoints[index] - position) <= radius:
        index += 1
    return index


def model_action(model, obs: torch.Tensor, state, deterministic: bool = True):
    if state is not None and hasattr(model, "act_recurrent"):
        with torch.no_grad():
            action, _, _, next_state = model.act_recurrent(obs, state, deterministic=deterministic)
        return action, next_state
    with torch.no_grad():
        action, _, _ = model.act(obs, deterministic=deterministic)
    return action, state


def collect_dagger_sequences(
    map_paths: list[str],
    train_cfg: dict,
    model,
    device: torch.device,
    rollouts_per_map: int,
    noise_std: float,
    expert_mix: float,
    seed_offset: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[dict]]:
    obs_sequences: list[torch.Tensor] = []
    action_sequences: list[torch.Tensor] = []
    metadata: list[dict] = []
    for map_path in map_paths:
        teacher_cfg = prepare_teacher_config(map_path, train_cfg)
        pure_cfg = load_task_config(map_path)
        pure_cfg["episode"]["max_steps"] = int(train_cfg.get("max_steps", 900))
        pure_cfg["goal"]["observation_max_distance"] = float(train_cfg.get("goal_observation_max", 10.0))
        waypoints = [np.array(point, dtype=np.float32) for point in teacher_cfg.get("map", {}).get("waypoints", [])]
        if not waypoints:
            waypoints = [np.array(pure_cfg["map"]["goal"], dtype=np.float32)]
        waypoint_radius = float(teacher_cfg.get("map", {}).get("waypoint_radius", 0.30))

        for episode_idx in range(rollouts_per_map):
            rng = np.random.default_rng(seed_offset + episode_idx)
            env = MujocoNavigationEnv(pure_cfg, num_envs=1, device=device, seed=seed_offset + episode_idx, auto_reset=False)
            obs = env.reset()
            state = model.initial_state(1, device) if hasattr(model, "initial_state") else None
            waypoint_index = waypoint_index_for_position(waypoints, env.envs[0].position, waypoint_radius)
            obs_list: list[np.ndarray] = []
            action_list: list[np.ndarray] = []
            final_info = None
            for _ in range(int(train_cfg.get("max_steps", 900))):
                single = env.envs[0]
                waypoint_index = waypoint_index_for_position(waypoints, single.position, waypoint_radius, waypoint_index)
                expert = teacher_action_to_goal(single, waypoints[waypoint_index])
                obs_list.append(single.observation_for_goal(single.final_goal))
                action_list.append(expert)
                policy_action, state = model_action(model, obs, state, deterministic=True)
                action_np = policy_action.detach().cpu().numpy()[0]
                mixed = (1.0 - expert_mix) * action_np + expert_mix * expert
                if noise_std > 0.0:
                    mixed += rng.normal(0.0, noise_std, size=2).astype(np.float32)
                mixed = np.clip(mixed, -1.0, 1.0).astype(np.float32)
                out = env.step(torch.as_tensor(mixed, dtype=torch.float32, device=device).view(1, 2))
                obs = out.observation
                final_info = out.info["raw"][0]
                if bool(out.done[0]):
                    break
            if len(obs_list) < 5:
                continue
            obs_sequences.append(torch.as_tensor(np.stack(obs_list), dtype=torch.float32, device=device))
            action_sequences.append(torch.as_tensor(np.stack(action_list), dtype=torch.float32, device=device))
            metadata.append(
                {
                    "map": pure_cfg["name"],
                    "steps": len(obs_list),
                    "success": bool(final_info and final_info["success"]),
                    "collision": bool(final_info and final_info["collision"]),
                    "timeout": bool(final_info and final_info["timeout"]),
                }
            )
    return obs_sequences, action_sequences, metadata


def sequence_loss(model, obs_seq: torch.Tensor, action_seq: torch.Tensor, train_cfg: dict) -> torch.Tensor:
    if isinstance(model, RecurrentActorCritic):
        mean, _ = model.forward_sequence(obs_seq.unsqueeze(0))
        pred = torch.tanh(mean.squeeze(0))
    else:
        mean, _ = model(obs_seq)
        pred = torch.tanh(mean)
    per_step = (pred - action_seq).pow(2).mean(dim=-1)
    weights = torch.ones_like(per_step)
    early_steps = int(train_cfg.get("early_weight_steps", 100))
    if early_steps > 0:
        weights[: min(early_steps, weights.numel())] *= float(train_cfg.get("early_weight", 4.0))
    near_ray = float(train_cfg.get("near_obstacle_ray", 0.22))
    near_weight = float(train_cfg.get("near_obstacle_weight", 5.0))
    if near_weight > 1.0:
        min_ray = torch.min(obs_seq[:, 6:], dim=-1).values
        weights = torch.where(min_ray < near_ray, weights * near_weight, weights)
    weights = weights / torch.clamp(weights.mean(), min=1e-6)
    return torch.mean(per_step * weights)


def evaluate_maps(map_paths: list[str], train_cfg: dict, model, run_dir: Path, suffix: str, device: torch.device) -> dict[str, float]:
    metrics_out: dict[str, float] = {}
    for map_path in map_paths:
        cfg = load_task_config(map_path)
        cfg["episode"]["max_steps"] = int(train_cfg.get("max_steps", 900))
        cfg["goal"]["observation_max_distance"] = float(train_cfg.get("goal_observation_max", 10.0))
        metrics, episodes = evaluate_policy(cfg, model, episodes=4, num_envs=1, device=device, deterministic=True)
        name = str(cfg["name"])
        out_prefix = run_dir / f"{name}_{suffix}"
        write_eval_outputs(metrics, episodes, out_prefix.with_suffix(".json"), out_prefix.with_suffix(".csv"))
        render_rollout_png(cfg, episodes[0], run_dir / f"{name}_{suffix}.png")
        render_rollout_gif(cfg, episodes[0], run_dir / f"{name}_{suffix}.gif")
        for key, value in metrics.items():
            metrics_out[f"{name}_{key}"] = value
    return metrics_out


def save_checkpoint(
    model,
    optimizer: torch.optim.Optimizer,
    run_dir: Path,
    epoch: int,
    policy_name: str,
    hidden_size: int,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": epoch,
            "policy": policy_name,
            "hidden_size": hidden_size,
            "training": "behavioral_cloning_from_astar_teacher",
            "eval_requires_astar": False,
        },
        run_dir / "latest.pt",
    )


def train_epochs(
    model,
    optimizer: torch.optim.Optimizer,
    obs_sequences: list[torch.Tensor],
    action_sequences: list[torch.Tensor],
    train_cfg: dict,
    run_dir: Path,
    policy_name: str,
    hidden_size: int,
    epochs: int,
    start_epoch: int,
    save_interval: int,
    label: str,
) -> list[dict]:
    history: list[dict] = []
    pbar = tqdm(range(epochs), desc=label)
    for local_epoch in pbar:
        losses = []
        order = torch.randperm(len(obs_sequences)).tolist()
        for idx in order:
            loss = sequence_loss(model, obs_sequences[idx], action_sequences[idx], train_cfg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch = start_epoch + local_epoch + 1
        row = {"epoch": epoch, "loss": float(np.mean(losses))}
        history.append(row)
        pbar.set_postfix(loss=f"{row['loss']:.5f}", sequences=len(obs_sequences))
        if save_interval > 0 and epoch % save_interval == 0:
            save_checkpoint(model, optimizer, run_dir, epoch, policy_name, hidden_size)
    return history


def main() -> None:
    args = parse_args()
    map_configs = resolve_map_configs(args)
    train_cfg = load_train_config(args.train_config)
    if args.policy is not None:
        train_cfg["policy"] = args.policy
    if args.hidden_size is not None:
        train_cfg["hidden_size"] = args.hidden_size
    if args.learning_rate is not None:
        train_cfg["learning_rate"] = args.learning_rate
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    device = torch.device(args.device)
    run_dir = ROOT / "results" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    obs_sequences, action_sequences, metadata = collect_sequences(map_configs, train_cfg, device)
    with (run_dir / "teacher_sequences.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["map", "steps", "success", "collision", "timeout"])
        writer.writeheader()
        writer.writerows(metadata)

    policy_name = str(train_cfg.get("policy", "gru"))
    hidden_size = int(train_cfg.get("hidden_size", 128))
    model = build_actor_critic(policy_name, 38, 2, hidden_size).to(device)
    optimizer_state = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        if checkpoint.get("policy", policy_name) != policy_name:
            raise ValueError(f"Checkpoint policy {checkpoint.get('policy')} does not match config policy {policy_name}.")
        model.load_state_dict(checkpoint["model_state"])
        optimizer_state = checkpoint.get("optimizer_state")

    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.get("learning_rate", 5e-4)))
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
        except ValueError:
            pass

    history = train_epochs(
        model,
        optimizer,
        obs_sequences,
        action_sequences,
        train_cfg,
        run_dir,
        policy_name,
        hidden_size,
        int(train_cfg.get("epochs", 500)),
        0,
        args.save_interval,
        "bc",
    )

    dagger_metadata: list[dict] = []
    total_epochs = int(train_cfg.get("epochs", 500))
    for iteration in range(args.dagger_iterations):
        new_obs, new_actions, new_meta = collect_dagger_sequences(
            map_configs,
            train_cfg,
            model,
            device,
            args.dagger_rollouts_per_map,
            args.dagger_noise,
            args.dagger_expert_mix,
            seed_offset=5000 + iteration * 1000,
        )
        obs_sequences.extend(new_obs)
        action_sequences.extend(new_actions)
        for row in new_meta:
            row["iteration"] = iteration + 1
        dagger_metadata.extend(new_meta)
        history.extend(
            train_epochs(
                model,
                optimizer,
                obs_sequences,
                action_sequences,
                train_cfg,
                run_dir,
                policy_name,
                hidden_size,
                args.dagger_epochs,
                total_epochs,
                args.save_interval,
                f"dagger{iteration + 1}",
            )
        )
        total_epochs += args.dagger_epochs

    save_checkpoint(model, optimizer, run_dir, total_epochs, policy_name, hidden_size)
    with (run_dir / "bc_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "loss"])
        writer.writeheader()
        writer.writerows(history)
    if dagger_metadata:
        with (run_dir / "dagger_sequences.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["iteration", "map", "steps", "success", "collision", "timeout"])
            writer.writeheader()
            writer.writerows(dagger_metadata)
    if not args.no_final_eval:
        print(evaluate_maps(map_configs, train_cfg, model, run_dir, "pure_eval", device))


if __name__ == "__main__":
    main()
