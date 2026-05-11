from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class ObstacleSpec:
    shape: str
    x: float
    y: float
    radius: float
    half_x: float
    half_y: float
    yaw: float = 0.0

    @property
    def center(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)


def local_box_point(point: np.ndarray, obs: ObstacleSpec) -> np.ndarray:
    dx = float(point[0]) - obs.x
    dy = float(point[1]) - obs.y
    c = cos(obs.yaw)
    s = sin(obs.yaw)
    return np.array([c * dx + s * dy, -s * dx + c * dy], dtype=np.float32)


def box_signed_distance(point: np.ndarray, obs: ObstacleSpec) -> float:
    local = local_box_point(point, obs)
    dx = abs(float(local[0])) - obs.half_x
    dy = abs(float(local[1])) - obs.half_y
    outside = np.array([max(dx, 0.0), max(dy, 0.0)], dtype=np.float32)
    return float(np.linalg.norm(outside) + min(max(dx, dy), 0.0))


def obstacle_count(config: dict, rng: np.random.Generator) -> int:
    low, high = config["obstacles"].get("count", [0, 0])
    return int(rng.integers(int(low), int(high) + 1))


def max_obstacles(config: dict) -> int:
    configured_max = int(config["obstacles"].get("count", [0, 0])[1])
    map_cfg = config.get("map", {})
    fixed_count = len(map_cfg.get("obstacles", [])) + len(map_cfg.get("dynamic_obstacles", []))
    return max(configured_max, fixed_count)


def has_fixed_map(config: dict) -> bool:
    map_cfg = config.get("map")
    return isinstance(map_cfg, dict) and bool(map_cfg.get("enabled", True)) and "start" in map_cfg and "goal" in map_cfg


def _map_obstacle_items(config: dict) -> list[dict[str, Any]]:
    map_cfg = config.get("map", {}) or {}
    return [*map_cfg.get("obstacles", []), *map_cfg.get("dynamic_obstacles", [])]


def fixed_obstacle_motions(config: dict) -> list[dict[str, Any]]:
    motions: list[dict[str, Any]] = []
    for item in _map_obstacle_items(config):
        motion = item.get("motion", {}) or {}
        motions.append(motion if isinstance(motion, dict) else {})
    return motions


def fixed_obstacles(config: dict) -> list[ObstacleSpec]:
    obstacles: list[ObstacleSpec] = []
    for item in _map_obstacle_items(config):
        shape = str(item.get("shape", "cylinder"))
        x = float(item["x"])
        y = float(item["y"])
        if shape == "box":
            half_x = float(item.get("half_x", item.get("width", 0.3) * 0.5))
            half_y = float(item.get("half_y", item.get("height", 0.3) * 0.5))
            radius = float(item.get("radius", max(half_x, half_y)))
            yaw = float(item.get("yaw", 0.0))
            obstacles.append(ObstacleSpec("box", x, y, radius, half_x, half_y, yaw))
        else:
            radius = float(item.get("radius", 0.2))
            obstacles.append(ObstacleSpec("cylinder", x, y, radius, radius, radius))
    return obstacles


def fixed_start_goal(config: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    map_cfg = config["map"]
    start_cfg = map_cfg["start"]
    goal_cfg = map_cfg["goal"]
    start = np.array([float(start_cfg[0]), float(start_cfg[1])], dtype=np.float32)
    yaw = float(start_cfg[2]) if len(start_cfg) >= 3 else 0.0
    goal = np.array([float(goal_cfg[0]), float(goal_cfg[1])], dtype=np.float32)
    jitter = map_cfg.get("jitter", {}) or {}
    if bool(jitter.get("enabled", False)):
        start += rng.normal(0.0, float(jitter.get("start_std", 0.0)), size=2).astype(np.float32)
        goal += rng.normal(0.0, float(jitter.get("goal_std", 0.0)), size=2).astype(np.float32)
        yaw += float(rng.normal(0.0, float(jitter.get("yaw_std", 0.0))))
    return start, goal, yaw


def random_obstacles(config: dict, rng: np.random.Generator) -> list[ObstacleSpec]:
    arena_half = float(config["arena"]["half_size"])
    count = obstacle_count(config, rng)
    radius_low, radius_high = config["obstacles"].get("radius", [0.1, 0.25])
    box_probability = float(config["obstacles"].get("box_probability", 0.0))
    min_clearance = float(config["obstacles"].get("min_clearance", 0.30))
    obstacles: list[ObstacleSpec] = []

    for _ in range(count):
        for _attempt in range(500):
            radius = float(rng.uniform(radius_low, radius_high))
            margin = radius + 0.35
            x = float(rng.uniform(-arena_half + margin, arena_half - margin))
            y = float(rng.uniform(-arena_half + margin, arena_half - margin))
            candidate = np.array([x, y], dtype=np.float32)
            if all(np.linalg.norm(candidate - obs.center) > radius + obs.radius + min_clearance for obs in obstacles):
                if rng.random() < box_probability:
                    half_x = float(radius * rng.uniform(0.75, 1.35))
                    half_y = float(radius * rng.uniform(0.75, 1.35))
                    obstacles.append(ObstacleSpec("box", x, y, radius, half_x, half_y, 0.0))
                else:
                    obstacles.append(ObstacleSpec("cylinder", x, y, radius, radius, radius))
                break
    return obstacles


def is_free(point: np.ndarray, obstacles: Iterable[ObstacleSpec], config: dict, padding: float = 0.0) -> bool:
    arena_half = float(config["arena"]["half_size"])
    robot_radius = float(config["robot"]["radius"]) + padding
    if np.any(np.abs(point[:2]) > arena_half - robot_radius):
        return False
    for obs in obstacles:
        if obs.shape == "box":
            if box_signed_distance(point, obs) <= robot_radius:
                return False
        else:
            if np.linalg.norm(point[:2] - obs.center) <= obs.radius + robot_radius:
                return False
    return True


def sample_free_point(
    obstacles: Iterable[ObstacleSpec],
    config: dict,
    rng: np.random.Generator,
    padding: float = 0.0,
) -> np.ndarray:
    arena_half = float(config["arena"]["half_size"])
    robot_radius = float(config["robot"]["radius"]) + padding
    for _ in range(2000):
        point = rng.uniform(-arena_half + robot_radius, arena_half - robot_radius, size=2).astype(np.float32)
        if is_free(point, obstacles, config, padding=padding):
            return point
    raise RuntimeError("Could not sample a free point in the MuJoCo navigation arena.")


def sample_start_goal(
    obstacles: Iterable[ObstacleSpec],
    config: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    min_distance = float(config["goal"]["min_distance"])
    max_distance = float(config["goal"]["max_distance"])
    for _ in range(2000):
        start = sample_free_point(obstacles, config, rng, padding=0.05)
        angle = float(rng.uniform(-np.pi, np.pi))
        distance = float(rng.uniform(min_distance, max_distance))
        goal = start + np.array([cos(angle), sin(angle)], dtype=np.float32) * distance
        if is_free(goal, obstacles, config, padding=0.05):
            yaw = float(rng.uniform(-np.pi, np.pi))
            return start.astype(np.float32), goal.astype(np.float32), yaw
    start = sample_free_point(obstacles, config, rng, padding=0.05)
    goal = sample_free_point(obstacles, config, rng, padding=0.05)
    yaw = float(rng.uniform(-np.pi, np.pi))
    return start.astype(np.float32), goal.astype(np.float32), yaw
