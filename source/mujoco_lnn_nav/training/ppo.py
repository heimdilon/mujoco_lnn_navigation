from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from mujoco_lnn_nav.envs.navigation import MujocoNavigationEnv
from mujoco_lnn_nav.models.policies import BaseActorCritic


@dataclass
class PPOConfig:
    total_steps: int = 100000
    num_envs: int = 16
    rollout_steps: int = 128
    minibatch_size: int = 256
    update_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4

    @classmethod
    def from_dict(cls, data: dict) -> "PPOConfig":
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in data.items() if key in fields})


def _checkpoint(path: Path, model: BaseActorCritic, optimizer: torch.optim.Optimizer, step: int, policy: str, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "policy": policy,
            "hidden_size": int(cfg.get("hidden_size", 128)),
        },
        path,
    )


def ppo_update(
    model: BaseActorCritic,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    old_values: torch.Tensor,
    cfg: PPOConfig,
) -> dict[str, float]:
    batch_size = obs.shape[0]
    indices = np.arange(batch_size)
    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    for _ in range(cfg.update_epochs):
        np.random.shuffle(indices)
        for start in range(0, batch_size, cfg.minibatch_size):
            mb = torch.as_tensor(indices[start : start + cfg.minibatch_size], dtype=torch.long, device=obs.device)
            log_prob, entropy, value = model.evaluate_actions(obs[mb], actions[mb])
            ratio = torch.exp(log_prob - old_log_probs[mb])
            policy_loss_1 = -advantages[mb] * ratio
            policy_loss_2 = -advantages[mb] * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
            policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
            value_clipped = old_values[mb] + torch.clamp(value - old_values[mb], -cfg.clip_coef, cfg.clip_coef)
            value_loss = 0.5 * torch.max((value - returns[mb]).pow(2), (value_clipped - returns[mb]).pow(2)).mean()
            entropy_loss = entropy.mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            metrics["policy_loss"] = float(policy_loss.detach().cpu())
            metrics["value_loss"] = float(value_loss.detach().cpu())
            metrics["entropy"] = float(entropy_loss.detach().cpu())
    return metrics


def collect_rollout(
    env: MujocoNavigationEnv,
    model: BaseActorCritic,
    obs: torch.Tensor,
    cfg: PPOConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
    device = obs.device
    obs_buf = torch.zeros((cfg.rollout_steps, env.num_envs, env.observation_dim), dtype=torch.float32, device=device)
    action_buf = torch.zeros((cfg.rollout_steps, env.num_envs, env.action_dim), dtype=torch.float32, device=device)
    logprob_buf = torch.zeros((cfg.rollout_steps, env.num_envs), dtype=torch.float32, device=device)
    reward_buf = torch.zeros((cfg.rollout_steps, env.num_envs), dtype=torch.float32, device=device)
    done_buf = torch.zeros((cfg.rollout_steps, env.num_envs), dtype=torch.float32, device=device)
    value_buf = torch.zeros((cfg.rollout_steps, env.num_envs), dtype=torch.float32, device=device)

    successes = 0
    collisions = 0
    timeouts = 0
    for step in range(cfg.rollout_steps):
        obs_buf[step] = obs
        with torch.no_grad():
            action, log_prob, value = model.act(obs)
        out = env.step(action)
        action_buf[step] = action
        logprob_buf[step] = log_prob
        reward_buf[step] = out.reward
        done_buf[step] = out.done.float()
        value_buf[step] = value
        successes += int(out.info["success"].sum().item())
        collisions += int(out.info["collision"].sum().item())
        timeouts += int(out.info["timeout"].sum().item())
        obs = out.observation

    with torch.no_grad():
        _, _, next_value = model.act(obs, deterministic=True)
    advantages = torch.zeros_like(reward_buf)
    last_gae = torch.zeros((env.num_envs,), dtype=torch.float32, device=device)
    for step in reversed(range(cfg.rollout_steps)):
        if step == cfg.rollout_steps - 1:
            next_non_terminal = 1.0 - done_buf[step]
            next_values = next_value
        else:
            next_non_terminal = 1.0 - done_buf[step]
            next_values = value_buf[step + 1]
        delta = reward_buf[step] + cfg.gamma * next_values * next_non_terminal - value_buf[step]
        last_gae = delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * last_gae
        advantages[step] = last_gae
    returns = advantages + value_buf

    flat = {
        "obs": obs_buf.reshape((-1, env.observation_dim)),
        "actions": action_buf.reshape((-1, env.action_dim)),
        "log_probs": logprob_buf.reshape(-1),
        "advantages": advantages.reshape(-1),
        "returns": returns.reshape(-1),
        "values": value_buf.reshape(-1),
    }
    stats = {
        "successes": float(successes),
        "collisions": float(collisions),
        "timeouts": float(timeouts),
        "reward_mean": float(reward_buf.mean().detach().cpu()),
    }
    return obs, flat, stats


def train_ppo(
    env: MujocoNavigationEnv,
    model: BaseActorCritic,
    cfg: PPOConfig,
    policy_name: str,
    run_dir: Path,
    train_cfg: dict,
    eval_fn: Callable[[BaseActorCritic, int], dict[str, float]] | None = None,
    eval_interval: int = 10000,
    start_step: int = 0,
    optimizer_state: dict | None = None,
) -> list[dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-5)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    obs = env.reset()
    global_step = int(start_step)
    history: list[dict[str, float]] = []
    pbar = tqdm(total=max(0, cfg.total_steps - global_step), desc=f"ppo-{policy_name}")
    next_eval = ((global_step // eval_interval) + 1) * eval_interval

    while global_step < cfg.total_steps:
        obs, rollout, rollout_stats = collect_rollout(env, model, obs, cfg)
        metrics = ppo_update(
            model,
            optimizer,
            rollout["obs"],
            rollout["actions"],
            rollout["log_probs"],
            rollout["advantages"],
            rollout["returns"],
            rollout["values"],
            cfg,
        )
        global_step += cfg.rollout_steps * env.num_envs
        pbar.update(min(cfg.rollout_steps * env.num_envs, cfg.total_steps - (global_step - cfg.rollout_steps * env.num_envs)))
        row = {"step": float(global_step), **rollout_stats, **metrics}
        history.append(row)
        _checkpoint(run_dir / "latest.pt", model, optimizer, global_step, policy_name, train_cfg)

        if eval_fn is not None and global_step >= next_eval:
            eval_metrics = eval_fn(model, global_step)
            row.update({f"eval_{key}": value for key, value in eval_metrics.items()})
            next_eval += eval_interval
    pbar.close()
    _checkpoint(run_dir / "latest.pt", model, optimizer, global_step, policy_name, train_cfg)
    return history
