from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, asdict
from math import pi
from typing import Any

import numpy as np

from mujoco_lnn_nav.envs.layouts import fixed_obstacles, is_free
from mujoco_lnn_nav.utils.map_generation import MapValidationResult, validate_map_config


@dataclass(frozen=True)
class MapAugmentationSettings:
    start_goal_jitter: float = 0.22
    yaw_jitter: float = 0.45
    obstacle_jitter: float = 0.07
    obstacle_scale_jitter: float = 0.04
    obstacle_yaw_jitter: float = 0.05
    max_attempts: int = 120
    validation_resolution: float = 0.16

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def build_augmented_map(
    base_config: dict[str, Any],
    map_name: str,
    seed: int,
    settings: MapAugmentationSettings | None = None,
) -> tuple[dict[str, Any], MapValidationResult]:
    settings = settings or MapAugmentationSettings()
    for attempt in range(settings.max_attempts):
        rng = np.random.default_rng(seed + attempt * 7919)
        cfg = _build_once(base_config, map_name, seed, attempt, settings, rng)
        result = validate_map_config(cfg, resolution=settings.validation_resolution)
        if result.valid:
            cfg["map"].setdefault("augmented", {})
            cfg["map"]["augmented"].update(
                {
                    "source": str(base_config.get("map", {}).get("name", base_config.get("name", "manual_map"))),
                    "seed": int(seed),
                    "attempt": int(attempt),
                    "validation_path_length": round(result.path_length, 4),
                    "validation_waypoint_count": int(result.waypoint_count),
                    **settings.to_dict(),
                }
            )
            return cfg, result
    raise RuntimeError(f"Could not create a valid augmented variant for {base_config.get('name', 'map')} after {settings.max_attempts} attempts.")


def _build_once(
    base_config: dict[str, Any],
    map_name: str,
    seed: int,
    attempt: int,
    settings: MapAugmentationSettings,
    rng: np.random.Generator,
) -> dict[str, Any]:
    cfg = deepcopy(base_config)
    cfg["name"] = map_name
    cfg["seed"] = int(seed + attempt)
    map_cfg = cfg.setdefault("map", {})
    source_name = str(map_cfg.get("name", base_config.get("name", "manual_map")))
    map_cfg["name"] = map_name
    map_cfg["base_map"] = source_name
    map_cfg["jitter"] = {"enabled": False, "start_std": 0.0, "goal_std": 0.0, "yaw_std": 0.0}

    map_cfg["obstacles"] = [
        _jitter_obstacle(item, idx, map_name, cfg, settings, rng) for idx, item in enumerate(map_cfg.get("obstacles", []))
    ]
    map_cfg["start"], map_cfg["goal"] = _jitter_start_goal(base_config, cfg, settings, rng)
    cfg.setdefault("obstacles", {})["count"] = [len(map_cfg["obstacles"]), len(map_cfg["obstacles"])]
    return cfg


def _jitter_start_goal(
    base_config: dict[str, Any],
    cfg: dict[str, Any],
    settings: MapAugmentationSettings,
    rng: np.random.Generator,
) -> tuple[list[float], list[float]]:
    base_map = base_config["map"]
    base_start = np.array(base_map["start"][:2], dtype=np.float32)
    base_goal = np.array(base_map["goal"][:2], dtype=np.float32)
    base_yaw = float(base_map["start"][2]) if len(base_map["start"]) >= 3 else 0.0
    obstacles = fixed_obstacles(cfg)

    start = _sample_near_free(base_start, obstacles, cfg, settings.start_goal_jitter, rng)
    goal = _sample_near_free(base_goal, obstacles, cfg, settings.start_goal_jitter, rng)
    yaw = _wrap_angle(base_yaw + float(rng.normal(0.0, settings.yaw_jitter)))
    return [float(start[0]), float(start[1]), yaw], [float(goal[0]), float(goal[1])]


def _sample_near_free(
    origin: np.ndarray,
    obstacles: list,
    cfg: dict[str, Any],
    std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    arena_half = float(cfg["arena"]["half_size"])
    robot_radius = float(cfg["robot"]["radius"])
    for _ in range(240):
        candidate = origin + rng.normal(0.0, std, size=2).astype(np.float32)
        candidate = np.clip(candidate, -arena_half + robot_radius + 0.05, arena_half - robot_radius - 0.05)
        if is_free(candidate, obstacles, cfg, padding=0.05):
            return candidate.astype(np.float32)
    if is_free(origin, obstacles, cfg, padding=0.05):
        return origin.astype(np.float32)
    raise RuntimeError("Could not sample a free jittered start/goal point.")


def _jitter_obstacle(
    item: dict[str, Any],
    index: int,
    map_name: str,
    cfg: dict[str, Any],
    settings: MapAugmentationSettings,
    rng: np.random.Generator,
) -> dict[str, Any]:
    updated = deepcopy(item)
    updated["id"] = f"{map_name}_obs_{index:03d}"
    radius = float(updated.get("radius", max(float(updated.get("half_x", 0.2)), float(updated.get("half_y", 0.2)))))
    arena_half = float(cfg["arena"]["half_size"])
    margin = radius + float(cfg["robot"]["radius"]) + 0.08
    updated["x"] = float(np.clip(float(updated["x"]) + rng.normal(0.0, settings.obstacle_jitter), -arena_half + margin, arena_half - margin))
    updated["y"] = float(np.clip(float(updated["y"]) + rng.normal(0.0, settings.obstacle_jitter), -arena_half + margin, arena_half - margin))

    scale = float(np.clip(1.0 + rng.normal(0.0, settings.obstacle_scale_jitter), 0.88, 1.12))
    if str(updated.get("shape", "cylinder")) == "box":
        updated["half_x"] = max(0.06, float(updated.get("half_x", radius)) * scale)
        updated["half_y"] = max(0.06, float(updated.get("half_y", radius)) * scale)
        updated["radius"] = max(updated["half_x"], updated["half_y"])
        updated["yaw"] = _wrap_angle(float(updated.get("yaw", 0.0)) + float(rng.normal(0.0, settings.obstacle_yaw_jitter)))
    else:
        updated["radius"] = max(0.06, radius * scale)
        updated["half_x"] = updated["radius"]
        updated["half_y"] = updated["radius"]
    return updated


def _wrap_angle(angle: float) -> float:
    return float((angle + pi) % (2.0 * pi) - pi)
