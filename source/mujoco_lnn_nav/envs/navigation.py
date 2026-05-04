from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from typing import Any

import mujoco
import numpy as np
import torch

from mujoco_lnn_nav.envs.layouts import (
    ObstacleSpec,
    box_signed_distance,
    fixed_obstacles,
    fixed_start_goal,
    has_fixed_map,
    is_free,
    max_obstacles,
    random_obstacles,
    sample_start_goal,
)
from mujoco_lnn_nav.envs.rays import cast_rays, wrap_angle
from mujoco_lnn_nav.envs.xml import build_navigation_xml


@dataclass
class StepOutput:
    observation: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    info: dict[str, Any]

    @property
    def done(self) -> torch.Tensor:
        return torch.logical_or(self.terminated, self.truncated)


class _SingleMujocoNavigationEnv:
    observation_dim = 38
    action_dim = 2

    def __init__(self, config: dict, seed: int = 0):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.max_obstacles = max_obstacles(config)
        self.model = mujoco.MjModel.from_xml_string(build_navigation_xml(config, self.max_obstacles))
        self.data = mujoco.MjData(self.model)
        self.obstacles: list[ObstacleSpec] = []
        self.goal = np.zeros(2, dtype=np.float32)
        self.final_goal = np.zeros(2, dtype=np.float32)
        self.waypoints: list[np.ndarray] = []
        self.waypoint_index = 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_distance = 0.0
        self.step_count = 0
        self.last_path: list[tuple[float, float]] = []
        self.last_yaws: list[float] = []
        self.last_distances: list[float] = []
        self.reset()

    @property
    def qpos(self) -> np.ndarray:
        return self.data.qpos[:3]

    @property
    def qvel(self) -> np.ndarray:
        return self.data.qvel[:3]

    @property
    def position(self) -> np.ndarray:
        return self.qpos[:2].astype(np.float32)

    @property
    def yaw(self) -> float:
        return float(wrap_angle(float(self.qpos[2])))

    def reset(self) -> np.ndarray:
        if has_fixed_map(self.config):
            self.obstacles = fixed_obstacles(self.config)
            start, goal, yaw = fixed_start_goal(self.config, self.rng)
        else:
            self.obstacles = random_obstacles(self.config, self.rng)
            start, goal, yaw = sample_start_goal(self.obstacles, self.config, self.rng)
        self.final_goal = goal.copy()
        self.waypoints = [np.array(point, dtype=np.float32) for point in self.config.get("map", {}).get("waypoints", [])]
        if self.waypoints:
            self.waypoint_index = 0
            while self.waypoint_index < len(self.waypoints) - 1:
                if np.linalg.norm(self.waypoints[self.waypoint_index] - start) > float(self.config.get("map", {}).get("waypoint_radius", 0.45)):
                    break
                self.waypoint_index += 1
            self.goal = self.waypoints[self.waypoint_index].copy()
        else:
            self.waypoint_index = 0
            self.goal = goal.copy()
        self.data.qpos[:3] = np.array([start[0], start[1], yaw], dtype=np.float64)
        self.data.qvel[:3] = 0.0
        self.data.ctrl[:] = 0.0
        self.prev_action[:] = 0.0
        self.step_count = 0
        self._apply_obstacle_geoms()
        self._apply_goal_marker()
        mujoco.mj_forward(self.model, self.data)
        self.prev_distance = self.goal_distance()
        self.last_path = [(float(start[0]), float(start[1]))]
        self.last_yaws = [float(self.yaw)]
        self.last_distances = [float(self.prev_distance)]
        return self.observation()

    def _geom_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)

    def _apply_obstacle_geoms(self) -> None:
        for idx in range(self.max_obstacles):
            cyl_id = self._geom_id(f"obstacle_cyl_{idx}")
            box_id = self._geom_id(f"obstacle_box_{idx}")
            for geom_id in (cyl_id, box_id):
                self.model.geom_pos[geom_id] = np.array([100.0, 100.0, -10.0], dtype=np.float64)
                self.model.geom_quat[geom_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
                self.model.geom_rgba[geom_id, 3] = 0.0
                self.model.geom_contype[geom_id] = 0
                self.model.geom_conaffinity[geom_id] = 0

            if idx >= len(self.obstacles):
                continue
            obs = self.obstacles[idx]
            if obs.shape == "box":
                self.model.geom_pos[box_id] = np.array([obs.x, obs.y, 0.35], dtype=np.float64)
                self.model.geom_size[box_id] = np.array([obs.half_x, obs.half_y, 0.35], dtype=np.float64)
                self.model.geom_quat[box_id] = np.array(
                    [cos(obs.yaw * 0.5), 0.0, 0.0, sin(obs.yaw * 0.5)], dtype=np.float64
                )
                self.model.geom_rgba[box_id] = np.array([0.76, 0.31, 0.12, 1.0], dtype=np.float64)
                self.model.geom_contype[box_id] = 1
                self.model.geom_conaffinity[box_id] = 1
            else:
                self.model.geom_pos[cyl_id] = np.array([obs.x, obs.y, 0.35], dtype=np.float64)
                self.model.geom_size[cyl_id] = np.array([obs.radius, 0.35, 0.0], dtype=np.float64)
                self.model.geom_rgba[cyl_id] = np.array([0.76, 0.18, 0.15, 1.0], dtype=np.float64)
                self.model.geom_contype[cyl_id] = 1
                self.model.geom_conaffinity[cyl_id] = 1

    def _apply_goal_marker(self) -> None:
        if self.model.nmocap > 0:
            self.data.mocap_pos[0] = np.array([self.goal[0], self.goal[1], 0.05], dtype=np.float64)

    def observation_for_goal(self, goal: np.ndarray) -> np.ndarray:
        goal_vec = goal.astype(np.float32) - self.position
        goal_distance = float(np.linalg.norm(goal_vec))
        goal_bearing = float(wrap_angle(np.arctan2(goal_vec[1], goal_vec[0]) - self.yaw))
        max_goal = float(self.config["goal"].get("observation_max_distance", self.config["sensors"]["max_range"]))
        max_linear = float(self.config["robot"]["max_linear_velocity"])
        max_angular = float(self.config["robot"]["max_angular_velocity"])
        num_rays = int(self.config["sensors"]["rays"])
        max_range = float(self.config["sensors"]["max_range"])
        rays = cast_rays(
            self.position,
            self.yaw,
            self.obstacles,
            num_rays,
            max_range,
            float(self.config["arena"]["half_size"]),
        )
        noise_std = float(self.config["sensors"].get("noise_std", 0.0))
        if noise_std > 0.0:
            rays = np.clip(rays + self.rng.normal(0.0, noise_std * max_range, size=rays.shape), 0.0, max_range)
        obs = np.concatenate(
            [
                np.array(
                    [
                        np.clip(goal_distance / max_goal, 0.0, 1.5),
                        goal_bearing / np.pi,
                        np.clip(float(self.qvel[0] * cos(self.yaw) + self.qvel[1] * sin(self.yaw)) / max_linear, -1.5, 1.5),
                        np.clip(float(self.qvel[2]) / max_angular, -1.5, 1.5),
                    ],
                    dtype=np.float32,
                ),
                self.prev_action.astype(np.float32),
                (rays / max_range).astype(np.float32),
            ]
        )
        if obs.shape[0] != self.observation_dim:
            raise RuntimeError(f"Observation dimension mismatch: {obs.shape[0]} != {self.observation_dim}")
        return obs.astype(np.float32)

    def observation(self) -> np.ndarray:
        return self.observation_for_goal(self.goal)

    def goal_distance(self) -> float:
        return float(np.linalg.norm(self.goal - self.position))

    def _collision(self) -> bool:
        pos = self.position
        if not is_free(pos, self.obstacles, self.config, padding=0.0):
            return True
        return False

    def _min_obstacle_distance(self) -> float:
        pos = self.position
        robot_radius = float(self.config["robot"]["radius"])
        best = float(self.config["sensors"]["max_range"])
        for obs in self.obstacles:
            if obs.shape == "box":
                best = min(best, box_signed_distance(pos, obs) - robot_radius)
            else:
                best = min(best, float(np.linalg.norm(pos - obs.center) - obs.radius - robot_radius))
        return best

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        max_linear = float(self.config["robot"]["max_linear_velocity"])
        max_angular = float(self.config["robot"]["max_angular_velocity"])
        linear = float(action[0] * max_linear)
        angular = float(action[1] * max_angular)
        yaw = self.yaw
        vx = linear * cos(yaw)
        vy = linear * sin(yaw)
        self.data.ctrl[:] = np.array([vx, vy, angular], dtype=np.float64)
        for _ in range(int(self.config.get("frame_skip", 4))):
            self.data.qvel[:3] = np.array([vx, vy, angular], dtype=np.float64)
            mujoco.mj_step(self.model, self.data)
        self.data.qpos[2] = float(wrap_angle(float(self.data.qpos[2])))
        self.step_count += 1

        distance = self.goal_distance()
        progress = self.prev_distance - distance
        self.prev_distance = distance
        waypoint_hit = False
        success = False
        waypoint_radius = float(self.config.get("map", {}).get("waypoint_radius", self.config["goal"]["radius"]))
        if self.waypoints and distance <= waypoint_radius:
            waypoint_hit = True
            if self.waypoint_index < len(self.waypoints) - 1:
                self.waypoint_index += 1
                self.goal = self.waypoints[self.waypoint_index].copy()
                self.prev_distance = self.goal_distance()
                distance = self.prev_distance
            else:
                success = self.goal_distance() <= float(self.config["goal"]["radius"]) or np.linalg.norm(self.final_goal - self.position) <= waypoint_radius
        else:
            success = distance <= float(self.config["goal"]["radius"])
        collision = self._collision()
        timeout = self.step_count >= int(self.config["episode"]["max_steps"])
        truncated = bool(timeout and not success and not collision)
        terminated = bool(success or collision)

        reward_cfg = self.config["reward"]
        heading = float(np.cos(float(wrap_angle(np.arctan2(self.goal[1] - self.position[1], self.goal[0] - self.position[0]) - self.yaw))))
        min_obstacle_distance = self._min_obstacle_distance()
        reward = (
            float(reward_cfg.get("progress_scale", 3.0)) * progress
            + float(reward_cfg.get("step_penalty", -0.01))
            + float(reward_cfg.get("heading_alignment_scale", 0.0)) * heading
            + float(reward_cfg.get("action_smoothness_penalty", -0.02)) * float(np.square(action - self.prev_action).sum())
        )
        if min_obstacle_distance < float(self.config["robot"]["radius"]) * 1.2:
            reward += float(reward_cfg.get("near_obstacle_penalty", -0.03))
        if waypoint_hit and not success:
            reward += float(reward_cfg.get("waypoint_bonus", 1.0))
        if success:
            reward += float(reward_cfg.get("success_bonus", 8.0))
        if collision:
            reward += float(reward_cfg.get("collision_penalty", -7.0))
        if truncated:
            reward += float(reward_cfg.get("timeout_penalty", -1.0))

        self.prev_action[:] = action
        obs = self.observation()
        self.last_path.append((float(self.position[0]), float(self.position[1])))
        self.last_yaws.append(float(self.yaw))
        self.last_distances.append(float(distance))
        info = {
            "success": success,
            "collision": collision,
            "timeout": truncated,
            "distance": distance,
            "progress": float(progress),
            "min_obstacle_distance": float(min_obstacle_distance),
            "waypoint_index": self.waypoint_index,
        }
        return obs, float(reward), terminated, truncated, info


class MujocoNavigationEnv:
    observation_dim = 38
    action_dim = 2

    def __init__(
        self,
        config: dict,
        num_envs: int | None = None,
        device: str | torch.device = "cpu",
        seed: int | None = None,
        auto_reset: bool = True,
    ):
        self.config = config
        self.num_envs = int(num_envs or config.get("num_envs", 1))
        self.device = torch.device(device)
        self.auto_reset = auto_reset
        base_seed = int(seed if seed is not None else config.get("seed", 0))
        self.envs = [_SingleMujocoNavigationEnv(config, seed=base_seed + idx * 9973) for idx in range(self.num_envs)]

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
            if self.auto_reset and (term or trunc):
                info["final_observation"] = obs
                info["final_path"] = list(env.last_path)
                info["final_yaws"] = list(env.last_yaws)
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
