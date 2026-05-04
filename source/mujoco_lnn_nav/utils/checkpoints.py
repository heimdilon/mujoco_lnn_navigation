from __future__ import annotations

from pathlib import Path

import torch

from mujoco_lnn_nav.models.policies import BaseActorCritic, build_actor_critic


def load_policy_from_checkpoint(path: str | Path, obs_dim: int, action_dim: int, device: str | torch.device = "cpu") -> BaseActorCritic:
    checkpoint = torch.load(Path(path), map_location=device)
    policy = checkpoint.get("policy", "mlp")
    hidden_size = int(checkpoint.get("hidden_size", 128))
    model = build_actor_critic(policy, obs_dim, action_dim, hidden_size=hidden_size).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model

