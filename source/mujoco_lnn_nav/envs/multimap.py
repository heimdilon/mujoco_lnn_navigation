from __future__ import annotations

from typing import Any

import numpy as np
import torch

from mujoco_lnn_nav.envs.navigation import StepOutput, _SingleMujocoNavigationEnv


class MultiMapNavigationEnv:
    observation_dim = 38
    action_dim = 2

    def __init__(
        self,
        configs: list[dict[str, Any]],
        num_envs: int,
        device: str | torch.device = "cpu",
        seed: int = 0,
        auto_reset: bool = True,
    ):
        if not configs:
            raise ValueError("At least one map config is required.")
        self.configs = configs
        self.config = configs[0]
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.auto_reset = auto_reset
        self.envs = [
            _SingleMujocoNavigationEnv(configs[idx % len(configs)], seed=seed + idx * 9973)
            for idx in range(self.num_envs)
        ]

    def reset(self) -> torch.Tensor:
        observations = [env.reset() for env in self.envs]
        return torch.as_tensor(np.stack(observations), dtype=torch.float32, device=self.device)

    def step(self, action: torch.Tensor | np.ndarray) -> StepOutput:
        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy()
        else:
            action_np = np.asarray(action, dtype=np.float32)
        if action_np.shape != (self.num_envs, self.action_dim):
            raise ValueError(f"Expected action shape {(self.num_envs, self.action_dim)}, got {action_np.shape}")

        observations: list[np.ndarray] = []
        rewards: list[float] = []
        terminated: list[bool] = []
        truncated: list[bool] = []
        infos: list[dict[str, Any]] = []
        for idx, env in enumerate(self.envs):
            obs, reward, term, trunc, info = env.step(action_np[idx])
            info["map_name"] = env.config.get("name", f"map_{idx}")
            if self.auto_reset and (term or trunc):
                info["final_observation"] = obs
                info["final_path"] = list(env.last_path)
                info["final_distances"] = list(env.last_distances)
                obs = env.reset()
            observations.append(obs)
            rewards.append(reward)
            terminated.append(term)
            truncated.append(trunc)
            infos.append(info)

        info_tensor = {
            "success": torch.as_tensor([info["success"] for info in infos], dtype=torch.bool, device=self.device),
            "collision": torch.as_tensor([info["collision"] for info in infos], dtype=torch.bool, device=self.device),
            "timeout": torch.as_tensor([info["timeout"] for info in infos], dtype=torch.bool, device=self.device),
            "distance": torch.as_tensor([info["distance"] for info in infos], dtype=torch.float32, device=self.device),
            "progress": torch.as_tensor([info["progress"] for info in infos], dtype=torch.float32, device=self.device),
            "min_obstacle_distance": torch.as_tensor(
                [info["min_obstacle_distance"] for info in infos], dtype=torch.float32, device=self.device
            ),
            "raw": infos,
        }
        return StepOutput(
            observation=torch.as_tensor(np.stack(observations), dtype=torch.float32, device=self.device),
            reward=torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
            terminated=torch.as_tensor(terminated, dtype=torch.bool, device=self.device),
            truncated=torch.as_tensor(truncated, dtype=torch.bool, device=self.device),
            info=info_tensor,
        )

