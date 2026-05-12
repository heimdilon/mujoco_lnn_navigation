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
from mujoco_lnn_nav.models.policies import build_actor_critic
from mujoco_lnn_nav.utils.evaluation import evaluate_policy, write_eval_outputs
from mujoco_lnn_nav.utils.planning import with_auto_waypoints
from mujoco_lnn_nav.utils.rendering import render_rollout_gif, render_rollout_png


def log_progress(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-configs", nargs="*", default=None)
    parser.add_argument("--split-config", default=None, help="Optional split YAML; train_maps are used for training.")
    parser.add_argument("--include-holdout-maps", action="store_true", help="Also train on holdout_maps from --split-config.")
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--policy",
        choices=[
            "mlp",
            "cfc",
            "lnn",
            "cfc_deep",
            "lnn_deep",
            "deep_lnn",
            "ncp",
            "ncp_cfc",
            "ncp_lnn",
            "gru",
            "lstm",
        ],
        default=None,
    )
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--batch-sequences", type=int, default=None)
    bucket_group = parser.add_mutually_exclusive_group()
    bucket_group.add_argument("--bucket-by-length", dest="bucket_by_length", action="store_true")
    bucket_group.add_argument("--no-bucket-by-length", dest="bucket_by_length", action="store_false")
    parser.set_defaults(bucket_by_length=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=20)
    parser.add_argument("--dagger-iterations", type=int, default=0)
    parser.add_argument("--dagger-rollouts-per-map", type=int, default=2)
    parser.add_argument("--dagger-epochs", type=int, default=40)
    parser.add_argument("--dagger-noise", type=float, default=0.06)
    parser.add_argument("--dagger-expert-mix", type=float, default=0.20)
    parser.add_argument("--log-interval", type=int, default=10, help="Print training loss every N epochs; 0 disables epoch logs.")
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
    episodes_per_map = int(train_cfg.get("episodes_per_map", 4))
    log_progress(f"[teacher] collecting A* teacher sequences maps={len(map_paths)} episodes_per_map={episodes_per_map}")
    for map_index, map_path in enumerate(map_paths, start=1):
        teacher_cfg = prepare_teacher_config(map_path, train_cfg)
        map_name = str(teacher_cfg["name"])
        map_meta_start = len(metadata)
        map_seq_start = len(obs_sequences)
        log_progress(f"[teacher] map {map_index}/{len(map_paths)} {map_name} start")
        for episode_idx in range(episodes_per_map):
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
        map_rows = metadata[map_meta_start:]
        successes = sum(1 for row in map_rows if row["success"])
        accepted = len(obs_sequences) - map_seq_start
        mean_steps = float(np.mean([row["steps"] for row in map_rows])) if map_rows else 0.0
        log_progress(
            f"[teacher] map {map_index}/{len(map_paths)} {map_name} done "
            f"success={successes}/{len(map_rows)} accepted={accepted} mean_steps={mean_steps:.1f}"
        )
    if not obs_sequences:
        raise RuntimeError("No successful teacher sequences were collected; cannot run behavioral cloning.")
    log_progress(f"[teacher] collected sequences={len(obs_sequences)} episodes={len(metadata)}")
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
    log_progress(
        f"[dagger] collecting rollouts maps={len(map_paths)} rollouts_per_map={rollouts_per_map} "
        f"noise={noise_std:.3f} expert_mix={expert_mix:.2f}"
    )
    for map_index, map_path in enumerate(map_paths, start=1):
        teacher_cfg = prepare_teacher_config(map_path, train_cfg)
        pure_cfg = load_task_config(map_path)
        pure_cfg["episode"]["max_steps"] = int(train_cfg.get("max_steps", 900))
        pure_cfg["goal"]["observation_max_distance"] = float(train_cfg.get("goal_observation_max", 10.0))
        waypoints = [np.array(point, dtype=np.float32) for point in teacher_cfg.get("map", {}).get("waypoints", [])]
        if not waypoints:
            waypoints = [np.array(pure_cfg["map"]["goal"], dtype=np.float32)]
        waypoint_radius = float(teacher_cfg.get("map", {}).get("waypoint_radius", 0.30))
        map_name = str(pure_cfg["name"])
        map_meta_start = len(metadata)
        map_seq_start = len(obs_sequences)
        log_progress(f"[dagger] map {map_index}/{len(map_paths)} {map_name} start")

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
        map_rows = metadata[map_meta_start:]
        successes = sum(1 for row in map_rows if row["success"])
        collisions = sum(1 for row in map_rows if row["collision"])
        accepted = len(obs_sequences) - map_seq_start
        mean_steps = float(np.mean([row["steps"] for row in map_rows])) if map_rows else 0.0
        log_progress(
            f"[dagger] map {map_index}/{len(map_paths)} {map_name} done "
            f"rollouts={len(map_rows)} success={successes} collision={collisions} "
            f"accepted={accepted} mean_steps={mean_steps:.1f}"
        )
    log_progress(f"[dagger] collected sequences={len(obs_sequences)} rollouts={len(metadata)}")
    return obs_sequences, action_sequences, metadata


def sequence_loss(model, obs_seq: torch.Tensor, action_seq: torch.Tensor, train_cfg: dict) -> torch.Tensor:
    if hasattr(model, "forward_sequence"):
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


def collate_sequence_batch(
    obs_sequences: list[torch.Tensor],
    action_sequences: list[torch.Tensor],
    indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(int(obs_sequences[idx].shape[0]) for idx in indices)
    obs_dim = int(obs_sequences[indices[0]].shape[-1])
    action_dim = int(action_sequences[indices[0]].shape[-1])
    device = obs_sequences[indices[0]].device
    obs_batch = torch.zeros((len(indices), max_len, obs_dim), dtype=torch.float32, device=device)
    action_batch = torch.zeros((len(indices), max_len, action_dim), dtype=torch.float32, device=device)
    valid = torch.zeros((len(indices), max_len), dtype=torch.float32, device=device)
    for batch_idx, seq_idx in enumerate(indices):
        seq_len = int(obs_sequences[seq_idx].shape[0])
        obs_batch[batch_idx, :seq_len] = obs_sequences[seq_idx]
        action_batch[batch_idx, :seq_len] = action_sequences[seq_idx]
        valid[batch_idx, :seq_len] = 1.0
    return obs_batch, action_batch, valid


def make_epoch_batches(
    obs_sequences: list[torch.Tensor],
    batch_size: int,
    bucket_by_length: bool,
) -> list[list[int]]:
    if not bucket_by_length:
        order = torch.randperm(len(obs_sequences)).tolist()
        return [order[start : start + batch_size] for start in range(0, len(order), batch_size)]

    sorted_indices = sorted(range(len(obs_sequences)), key=lambda idx: int(obs_sequences[idx].shape[0]))
    batches = [sorted_indices[start : start + batch_size] for start in range(0, len(sorted_indices), batch_size)]
    shuffled_batches = []
    for batch_idx in torch.randperm(len(batches)).tolist():
        batch = batches[batch_idx]
        if len(batch) > 1:
            perm = torch.randperm(len(batch)).tolist()
            batch = [batch[idx] for idx in perm]
        shuffled_batches.append(batch)
    return shuffled_batches


def sequence_batch_loss(
    model,
    obs_batch: torch.Tensor,
    action_batch: torch.Tensor,
    valid: torch.Tensor,
    train_cfg: dict,
) -> torch.Tensor:
    if hasattr(model, "forward_sequence"):
        mean, _ = model.forward_sequence(obs_batch)
    else:
        flat_obs = obs_batch.reshape(-1, obs_batch.shape[-1])
        flat_mean, _ = model(flat_obs)
        mean = flat_mean.reshape(obs_batch.shape[0], obs_batch.shape[1], -1)
    pred = torch.tanh(mean)
    per_step = (pred - action_batch).pow(2).mean(dim=-1)
    weights = torch.ones_like(per_step)
    early_steps = int(train_cfg.get("early_weight_steps", 100))
    if early_steps > 0:
        weights[:, : min(early_steps, weights.shape[1])] *= float(train_cfg.get("early_weight", 4.0))
    near_ray = float(train_cfg.get("near_obstacle_ray", 0.22))
    near_weight = float(train_cfg.get("near_obstacle_weight", 5.0))
    if near_weight > 1.0:
        min_ray = torch.min(obs_batch[:, :, 6:], dim=-1).values
        weights = torch.where(min_ray < near_ray, weights * near_weight, weights)
    valid_count = torch.clamp(valid.sum(), min=1.0)
    mean_weight = torch.clamp((weights * valid).sum() / valid_count, min=1e-6)
    weights = weights / mean_weight
    return ((per_step * weights * valid).sum() / valid_count)


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
    model_kwargs: dict | None = None,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": epoch,
            "policy": policy_name,
            "policy_impl": getattr(model, "policy_impl", policy_name),
            "hidden_size": hidden_size,
            "model_kwargs": dict(model_kwargs or {}),
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
    model_kwargs: dict | None,
    epochs: int,
    start_epoch: int,
    save_interval: int,
    label: str,
    log_interval: int,
) -> list[dict]:
    history: list[dict] = []
    batch_size = max(1, int(train_cfg.get("batch_sequences", 1)))
    log_progress(
        f"[{label}] training start epochs={epochs} start_epoch={start_epoch} "
        f"sequences={len(obs_sequences)} batch_sequences={batch_size} "
        f"bucket_by_length={bool(train_cfg.get('bucket_by_length', False))}"
    )
    pbar = tqdm(range(epochs), desc=label)
    for local_epoch in pbar:
        losses = []
        batches = make_epoch_batches(
            obs_sequences,
            batch_size,
            bucket_by_length=bool(train_cfg.get("bucket_by_length", False)),
        )
        for batch_indices in batches:
            if len(batch_indices) == 1:
                idx = batch_indices[0]
                loss = sequence_loss(model, obs_sequences[idx], action_sequences[idx], train_cfg)
            else:
                obs_batch, action_batch, valid = collate_sequence_batch(obs_sequences, action_sequences, batch_indices)
                loss = sequence_batch_loss(model, obs_batch, action_batch, valid, train_cfg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch = start_epoch + local_epoch + 1
        row = {"epoch": epoch, "loss": float(np.mean(losses))}
        history.append(row)
        pbar.set_postfix(loss=f"{row['loss']:.5f}", sequences=len(obs_sequences))
        should_log = (
            log_interval > 0
            and (local_epoch == 0 or (local_epoch + 1) % log_interval == 0 or local_epoch + 1 == epochs)
        )
        if should_log:
            log_progress(
                f"[{label}] epoch {local_epoch + 1}/{epochs} global_epoch={epoch} "
                f"loss={row['loss']:.5f} sequences={len(obs_sequences)} batches={len(batches)}"
            )
        if save_interval > 0 and epoch % save_interval == 0:
            save_checkpoint(model, optimizer, run_dir, epoch, policy_name, hidden_size, model_kwargs)
            log_progress(f"[checkpoint] saved latest.pt global_epoch={epoch}")
    if epochs == 0:
        log_progress(f"[{label}] skipped epochs=0")
    else:
        log_progress(f"[{label}] training done final_epoch={start_epoch + epochs} final_loss={history[-1]['loss']:.5f}")
    return history


def main() -> None:
    args = parse_args()
    map_configs = resolve_map_configs(args)
    train_cfg = load_train_config(args.train_config)
    if args.policy is not None:
        train_cfg["policy"] = args.policy
    if args.hidden_size is not None:
        train_cfg["hidden_size"] = args.hidden_size
    if args.batch_sequences is not None:
        train_cfg["batch_sequences"] = args.batch_sequences
    if args.bucket_by_length is not None:
        train_cfg["bucket_by_length"] = args.bucket_by_length
    if args.learning_rate is not None:
        train_cfg["learning_rate"] = args.learning_rate
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    device = torch.device(args.device)
    run_dir = ROOT / "results" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_progress(f"[run] name={args.run_name}")
    log_progress(f"[run] train_config={args.train_config}")
    log_progress(f"[run] maps={len(map_configs)} device={device} output={run_dir}")
    log_progress(
        f"[run] epochs={train_cfg.get('epochs', 500)} dagger_iterations={args.dagger_iterations} "
        f"dagger_epochs={args.dagger_epochs} batch_sequences={train_cfg.get('batch_sequences', 1)}"
    )

    obs_sequences, action_sequences, metadata = collect_sequences(map_configs, train_cfg, device)
    with (run_dir / "teacher_sequences.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["map", "steps", "success", "collision", "timeout"])
        writer.writeheader()
        writer.writerows(metadata)
    log_progress(f"[teacher] metadata saved {run_dir / 'teacher_sequences.csv'}")

    policy_name = str(train_cfg.get("policy", "gru"))
    hidden_size = int(train_cfg.get("hidden_size", 128))
    model_kwargs = dict(train_cfg.get("model_kwargs") or {})
    model = build_actor_critic(policy_name, 38, 2, hidden_size, **model_kwargs).to(device)
    n_params = sum(param.numel() for param in model.parameters())
    log_progress(
        f"[model] policy={policy_name} impl={getattr(model, 'policy_impl', policy_name)} "
        f"hidden_size={hidden_size} params={n_params} model_kwargs={model_kwargs}"
    )
    optimizer_state = None
    resume_epoch = 0
    if args.resume:
        log_progress(f"[resume] loading {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        if checkpoint.get("policy", policy_name) != policy_name:
            raise ValueError(f"Checkpoint policy {checkpoint.get('policy')} does not match config policy {policy_name}.")
        model.load_state_dict(checkpoint["model_state"])
        optimizer_state = checkpoint.get("optimizer_state")
        resume_epoch = int(checkpoint.get("step", 0) or 0)
        log_progress(f"[resume] loaded step={resume_epoch}")

    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.get("learning_rate", 5e-4)))
    log_progress(f"[optimizer] Adam lr={float(train_cfg.get('learning_rate', 5e-4))}")
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            log_progress("[optimizer] restored state")
        except ValueError:
            log_progress("[optimizer] checkpoint optimizer state incompatible; using fresh optimizer")
            pass

    requested_epochs = int(train_cfg.get("epochs", 500))
    history = train_epochs(
        model,
        optimizer,
        obs_sequences,
        action_sequences,
        train_cfg,
        run_dir,
        policy_name,
        hidden_size,
        model_kwargs,
        requested_epochs,
        resume_epoch,
        args.save_interval,
        "bc",
        args.log_interval,
    )

    dagger_metadata: list[dict] = []
    total_epochs = resume_epoch + requested_epochs
    for iteration in range(args.dagger_iterations):
        log_progress(f"[dagger{iteration + 1}] iteration start current_total_epoch={total_epochs}")
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
        log_progress(
            f"[dagger{iteration + 1}] appended sequences={len(new_obs)} "
            f"total_sequences={len(obs_sequences)}"
        )
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
                model_kwargs,
                args.dagger_epochs,
                total_epochs,
                args.save_interval,
                f"dagger{iteration + 1}",
                args.log_interval,
            )
        )
        total_epochs += args.dagger_epochs

    save_checkpoint(model, optimizer, run_dir, total_epochs, policy_name, hidden_size, model_kwargs)
    log_progress(f"[checkpoint] saved final latest.pt global_epoch={total_epochs}")
    with (run_dir / "bc_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "loss"])
        writer.writeheader()
        writer.writerows(history)
    log_progress(f"[history] saved {run_dir / 'bc_history.csv'} rows={len(history)}")
    if dagger_metadata:
        with (run_dir / "dagger_sequences.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["iteration", "map", "steps", "success", "collision", "timeout"])
            writer.writeheader()
            writer.writerows(dagger_metadata)
        log_progress(f"[dagger] metadata saved {run_dir / 'dagger_sequences.csv'} rows={len(dagger_metadata)}")
    if not args.no_final_eval:
        log_progress("[eval] final pure policy eval start")
        print(evaluate_maps(map_configs, train_cfg, model, run_dir, "pure_eval", device), flush=True)
        log_progress("[eval] final pure policy eval done")
    log_progress("[run] done")


if __name__ == "__main__":
    main()
