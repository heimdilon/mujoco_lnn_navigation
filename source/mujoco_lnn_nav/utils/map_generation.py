from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

from mujoco_lnn_nav.envs.layouts import ObstacleSpec, fixed_obstacles, is_free
from mujoco_lnn_nav.utils.planning import plan_waypoints


MAP_TYPES = ("easy_open", "clutter", "wall_gap", "zigzag", "corridor", "u_trap")
DIFFICULTIES = ("easy", "medium", "hard")

_DIFFICULTY_SETTINGS = {
    "easy": {"gap": 1.45, "random": 4, "radius": (0.14, 0.22), "wall_thickness": 0.12},
    "medium": {"gap": 1.15, "random": 7, "radius": (0.16, 0.26), "wall_thickness": 0.14},
    "hard": {"gap": 0.90, "random": 10, "radius": (0.18, 0.30), "wall_thickness": 0.16},
}


@dataclass(frozen=True)
class MapValidationResult:
    valid: bool
    reason: str
    path_length: float = 0.0
    waypoint_count: int = 0


def build_generated_map(
    base_config: dict[str, Any],
    map_name: str,
    map_type: str,
    seed: int,
    difficulty: str = "medium",
    max_attempts: int = 80,
    validation_resolution: float = 0.16,
) -> dict[str, Any]:
    if map_type not in MAP_TYPES:
        raise ValueError(f"Unknown map type '{map_type}'. Expected one of: {', '.join(MAP_TYPES)}")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"Unknown difficulty '{difficulty}'. Expected one of: {', '.join(DIFFICULTIES)}")

    for attempt in range(max_attempts):
        rng = np.random.default_rng(seed + attempt * 9973)
        cfg = _build_once(base_config, map_name, map_type, seed, attempt, difficulty, rng)
        result = validate_map_config(cfg, resolution=validation_resolution)
        if result.valid:
            cfg["map"].setdefault("generated", {})
            cfg["map"]["generated"].update(
                {
                    "type": map_type,
                    "difficulty": difficulty,
                    "seed": int(seed),
                    "attempt": int(attempt),
                    "validation_path_length": round(result.path_length, 4),
                    "validation_waypoint_count": int(result.waypoint_count),
                }
            )
            return cfg
    raise RuntimeError(f"Could not generate a valid {difficulty} {map_type} map after {max_attempts} attempts.")


def validate_map_config(cfg: dict[str, Any], resolution: float = 0.16) -> MapValidationResult:
    map_cfg = cfg.get("map", {})
    if not map_cfg.get("enabled", False):
        return MapValidationResult(False, "map.disabled")
    if "start" not in map_cfg or "goal" not in map_cfg:
        return MapValidationResult(False, "missing.start_or_goal")

    obstacles = fixed_obstacles(cfg)
    start = np.array(map_cfg["start"][:2], dtype=np.float32)
    goal = np.array(map_cfg["goal"][:2], dtype=np.float32)
    if not is_free(start, obstacles, cfg, padding=0.05):
        return MapValidationResult(False, "start.not_free")
    if not is_free(goal, obstacles, cfg, padding=0.05):
        return MapValidationResult(False, "goal.not_free")

    min_distance = float(cfg.get("goal", {}).get("min_distance", 0.0))
    if float(np.linalg.norm(goal - start)) < min_distance:
        return MapValidationResult(False, "start_goal.too_close")

    try:
        waypoints = plan_waypoints(cfg, resolution=resolution, simplify=True)
    except Exception as exc:
        return MapValidationResult(False, f"path.not_found:{exc}")

    path = [start.tolist()] + waypoints
    length = _polyline_length(path)
    if length <= 0.0:
        return MapValidationResult(False, "path.empty")
    return MapValidationResult(True, "ok", length, len(waypoints))


def build_map_gallery_image(configs: Iterable[dict[str, Any]], columns: int = 3, cell_size: int = 360) -> Image.Image:
    configs = list(configs)
    if not configs:
        raise ValueError("At least one map config is required to render a gallery.")
    columns = max(1, int(columns))
    rows = int(np.ceil(len(configs) / columns))
    image = Image.new("RGB", (columns * cell_size, rows * cell_size), (246, 247, 244))
    for idx, cfg in enumerate(configs):
        cell = render_map_preview(cfg, size=cell_size)
        x = (idx % columns) * cell_size
        y = (idx // columns) * cell_size
        image.paste(cell, (x, y))
    return image


def render_map_gallery(configs: Iterable[dict[str, Any]], path: str | Path, columns: int = 3, cell_size: int = 360) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = build_map_gallery_image(configs, columns=columns, cell_size=cell_size)
    image.save(path)


def render_map_preview(cfg: dict[str, Any], size: int = 360) -> Image.Image:
    margin = max(22, size // 14)
    image = Image.new("RGB", (size, size), (246, 247, 244))
    draw = ImageDraw.Draw(image)
    world_to_px = _world_to_px_factory(cfg, size, margin)
    draw.rectangle([margin, margin, size - margin, size - margin], outline=(70, 76, 82), width=2)

    for obs in cfg.get("map", {}).get("obstacles", []):
        _draw_obstacle(draw, world_to_px, obs, cfg, size, margin)

    map_cfg = cfg["map"]
    path = [map_cfg["start"][:2]]
    try:
        path.extend(plan_waypoints(cfg, resolution=0.18, simplify=True))
    except Exception:
        path.append(map_cfg["goal"][:2])
    px_path = [world_to_px(point) for point in path]
    if len(px_path) > 1:
        draw.line(px_path, fill=(48, 103, 180), width=max(2, size // 120))

    sx, sy = world_to_px(map_cfg["start"][:2])
    gx, gy = world_to_px(map_cfg["goal"][:2])
    marker = max(5, size // 44)
    draw.ellipse([sx - marker, sy - marker, sx + marker, sy + marker], fill=(250, 198, 64), outline=(120, 90, 20))
    draw.ellipse([gx - marker, gy - marker, gx + marker, gy + marker], fill=(39, 155, 78), outline=(20, 90, 45))
    label = str(map_cfg.get("name", cfg.get("name", "map")))[:34]
    draw.rectangle([0, 0, size, margin - 4], fill=(246, 247, 244))
    draw.text((8, 6), label, fill=(35, 39, 44))
    return image


def _build_once(
    base_config: dict[str, Any],
    map_name: str,
    map_type: str,
    seed: int,
    attempt: int,
    difficulty: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    settings = _DIFFICULTY_SETTINGS[difficulty]
    cfg = deepcopy(base_config)
    cfg["name"] = map_name
    cfg["seed"] = int(seed + attempt)
    cfg.setdefault("arena", {})["half_size"] = float(cfg.get("arena", {}).get("half_size", 4.0))
    cfg.setdefault("map", {})

    if map_type == "easy_open":
        start, goal, obstacles = _easy_open(cfg, settings, rng)
    elif map_type == "clutter":
        start, goal, obstacles = _clutter(cfg, settings, rng)
    elif map_type == "wall_gap":
        start, goal, obstacles = _wall_gap(cfg, settings, rng)
    elif map_type == "zigzag":
        start, goal, obstacles = _zigzag(cfg, settings, rng)
    elif map_type == "corridor":
        start, goal, obstacles = _corridor(cfg, settings, rng)
    elif map_type == "u_trap":
        start, goal, obstacles = _u_trap(cfg, settings, rng)
    else:
        raise AssertionError(map_type)

    cfg.setdefault("obstacles", {})["count"] = [len(obstacles), len(obstacles)]
    cfg["map"] = {
        "enabled": True,
        "name": map_name,
        "base_task": str(base_config.get("name", "open_clutter")),
        "start": [float(start[0]), float(start[1]), float(start[2])],
        "goal": [float(goal[0]), float(goal[1])],
        "jitter": {"enabled": False, "start_std": 0.0, "goal_std": 0.0, "yaw_std": 0.0},
        "obstacles": obstacles,
    }
    return cfg


def _easy_open(cfg: dict[str, Any], settings: dict[str, Any], rng: np.random.Generator) -> tuple[list[float], list[float], list[dict[str, Any]]]:
    start = [-3.15, -3.15, 0.0]
    goal = [3.15, 3.15]
    obstacles: list[dict[str, Any]] = []
    _add_random_obstacles(cfg, obstacles, start, goal, rng, max(2, settings["random"] - 2), settings)
    if not obstacles:
        obstacles.append(_cylinder("easy_marker_0", -1.4, 1.8, 0.18))
    return start, goal, obstacles


def _clutter(cfg: dict[str, Any], settings: dict[str, Any], rng: np.random.Generator) -> tuple[list[float], list[float], list[dict[str, Any]]]:
    start = [-3.2, -3.1, 0.0]
    goal = [3.15, 3.1]
    obstacles: list[dict[str, Any]] = []
    _add_random_obstacles(cfg, obstacles, start, goal, rng, settings["random"] + 4, settings)
    return start, goal, obstacles


def _wall_gap(cfg: dict[str, Any], settings: dict[str, Any], rng: np.random.Generator) -> tuple[list[float], list[float], list[dict[str, Any]]]:
    start = [-3.2, -3.15, 0.0]
    goal = [3.15, 3.15]
    y = float(rng.uniform(-0.45, 0.45))
    gap = float(settings["gap"])
    gap_center = float(rng.uniform(-0.9, 0.9))
    obstacles = _split_horizontal_wall("wall_gap", y, gap_center, gap, float(settings["wall_thickness"]))
    _add_random_obstacles(cfg, obstacles, start, goal, rng, max(0, settings["random"] - 4), settings)
    return start, goal, obstacles


def _zigzag(cfg: dict[str, Any], settings: dict[str, Any], rng: np.random.Generator) -> tuple[list[float], list[float], list[dict[str, Any]]]:
    start = [-3.25, -3.0, 0.0]
    goal = [3.25, 3.0]
    extent = 3.45
    gap = float(settings["gap"]) + 0.35
    thickness = float(settings["wall_thickness"])
    obstacles: list[dict[str, Any]] = []
    for idx, x in enumerate([-2.25, -0.9, 0.45, 1.8]):
        if idx % 2 == 0:
            y0, y1 = -extent, extent - gap
        else:
            y0, y1 = -extent + gap, extent
        obstacles.append(_wall_between(f"zigzag_{idx}", (x, y0), (x, y1), thickness))
    _add_random_obstacles(cfg, obstacles, start, goal, rng, max(0, settings["random"] - 6), settings)
    return start, goal, obstacles


def _corridor(cfg: dict[str, Any], settings: dict[str, Any], rng: np.random.Generator) -> tuple[list[float], list[float], list[dict[str, Any]]]:
    start = [-3.05, 0.0, 0.0]
    goal = [3.05, 0.0]
    thickness = float(settings["wall_thickness"])
    gap = float(settings["gap"])
    obstacles = [
        _wall_between("corridor_top", (-3.2, 1.25), (3.2, 1.25), thickness),
        _wall_between("corridor_bottom", (-3.2, -1.25), (3.2, -1.25), thickness),
    ]
    for idx, x in enumerate([-1.65, -0.25, 1.15]):
        if idx % 2 == 0:
            obstacles.append(_wall_between(f"corridor_gate_{idx}", (x, -1.25), (x, 1.25 - gap), thickness))
        else:
            obstacles.append(_wall_between(f"corridor_gate_{idx}", (x, -1.25 + gap), (x, 1.25), thickness))
    _add_random_obstacles(cfg, obstacles, start, goal, rng, max(0, settings["random"] - 7), settings)
    return start, goal, obstacles


def _u_trap(cfg: dict[str, Any], settings: dict[str, Any], rng: np.random.Generator) -> tuple[list[float], list[float], list[dict[str, Any]]]:
    start = [0.0, 0.35, -pi / 2.0]
    goal = [3.15, -3.1]
    thickness = float(settings["wall_thickness"])
    arm = 1.05 + (0.10 if settings["gap"] < 1.0 else 0.0)
    top = 1.55
    bottom = -1.0
    obstacles = [
        _wall_between("u_left", (-arm, bottom), (-arm, top), thickness),
        _wall_between("u_right", (arm, bottom), (arm, top), thickness),
        _wall_between("u_top", (-arm, top), (arm, top), thickness),
    ]
    _add_random_obstacles(cfg, obstacles, start, goal, rng, max(0, settings["random"] - 5), settings)
    return start, goal, obstacles


def _split_horizontal_wall(prefix: str, y: float, gap_center: float, gap: float, thickness: float) -> list[dict[str, Any]]:
    extent = 3.55
    gap_left = gap_center - gap * 0.5
    gap_right = gap_center + gap * 0.5
    segments: list[dict[str, Any]] = []
    if gap_left > -extent:
        segments.append(_wall_between(f"{prefix}_left", (-extent, y), (gap_left, y), thickness))
    if gap_right < extent:
        segments.append(_wall_between(f"{prefix}_right", (gap_right, y), (extent, y), thickness))
    return segments


def _add_random_obstacles(
    cfg: dict[str, Any],
    obstacles: list[dict[str, Any]],
    start: list[float],
    goal: list[float],
    rng: np.random.Generator,
    count: int,
    settings: dict[str, Any],
) -> None:
    arena_half = float(cfg["arena"]["half_size"])
    radius_low, radius_high = settings["radius"]
    start_point = np.array(start[:2], dtype=np.float32)
    goal_point = np.array(goal[:2], dtype=np.float32)
    for idx in range(count):
        for _attempt in range(500):
            radius = float(rng.uniform(radius_low, radius_high))
            margin = radius + 0.45
            x = float(rng.uniform(-arena_half + margin, arena_half - margin))
            y = float(rng.uniform(-arena_half + margin, arena_half - margin))
            center = np.array([x, y], dtype=np.float32)
            if np.linalg.norm(center - start_point) < 0.8 or np.linalg.norm(center - goal_point) < 0.8:
                continue
            if _distance_to_segment(center, start_point, goal_point) < 0.38 and rng.random() < 0.65:
                continue
            item = (
                _box(f"box_{idx}", x, y, radius * rng.uniform(0.75, 1.45), radius * rng.uniform(0.75, 1.30), 0.0)
                if rng.random() < 0.35
                else _cylinder(f"cylinder_{idx}", x, y, radius)
            )
            specs = [_spec_from_item(obs) for obs in obstacles + [item]]
            if is_free(start_point, specs, cfg, padding=0.05) and is_free(goal_point, specs, cfg, padding=0.05):
                obstacles.append(item)
                break


def _wall_between(id_: str, a: tuple[float, float], b: tuple[float, float], thickness: float) -> dict[str, Any]:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    length = hypot(bx - ax, by - ay)
    yaw = atan2(by - ay, bx - ax)
    return _box(id_, (ax + bx) * 0.5, (ay + by) * 0.5, length * 0.5, thickness * 0.5, yaw, "wall")


def _box(id_: str, x: float, y: float, half_x: float, half_y: float, yaw: float, kind: str = "box") -> dict[str, Any]:
    return {
        "id": id_,
        "shape": "box",
        "x": float(x),
        "y": float(y),
        "half_x": float(half_x),
        "half_y": float(half_y),
        "radius": float(max(half_x, half_y)),
        "yaw": float(yaw),
        "kind": kind,
    }


def _cylinder(id_: str, x: float, y: float, radius: float) -> dict[str, Any]:
    return {
        "id": id_,
        "shape": "cylinder",
        "x": float(x),
        "y": float(y),
        "radius": float(radius),
        "half_x": float(radius),
        "half_y": float(radius),
    }


def _spec_from_item(item: dict[str, Any]) -> ObstacleSpec:
    if item["shape"] == "box":
        return ObstacleSpec(
            "box",
            float(item["x"]),
            float(item["y"]),
            float(item["radius"]),
            float(item["half_x"]),
            float(item["half_y"]),
            float(item.get("yaw", 0.0)),
        )
    radius = float(item["radius"])
    return ObstacleSpec("cylinder", float(item["x"]), float(item["y"]), radius, radius, radius)


def _distance_to_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-8:
        return float(np.linalg.norm(point - a))
    t = max(0.0, min(1.0, float(np.dot(point - a, ab) / denom)))
    projection = a + t * ab
    return float(np.linalg.norm(point - projection))


def _polyline_length(points: list[list[float]]) -> float:
    total = 0.0
    for a, b in zip(points, points[1:]):
        total += hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
    return total


def _world_to_px_factory(config: dict[str, Any], size: int, margin: int):
    arena_half = float(config["arena"]["half_size"])

    def world_to_px(point: tuple[float, float] | list[float]) -> tuple[int, int]:
        x = margin + (float(point[0]) + arena_half) / (2.0 * arena_half) * (size - 2 * margin)
        y = size - margin - (float(point[1]) + arena_half) / (2.0 * arena_half) * (size - 2 * margin)
        return int(round(x)), int(round(y))

    return world_to_px


def _draw_obstacle(draw: ImageDraw.ImageDraw, world_to_px, obs: dict[str, Any], cfg: dict[str, Any], size: int, margin: int) -> None:
    arena_half = float(cfg["arena"]["half_size"])
    if obs["shape"] == "box":
        yaw = float(obs.get("yaw", 0.0))
        hx = float(obs["half_x"])
        hy = float(obs["half_y"])
        c = cos(yaw)
        s = sin(yaw)
        corners = []
        for lx, ly in [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]:
            wx = float(obs["x"]) + c * lx - s * ly
            wy = float(obs["y"]) + s * lx + c * ly
            corners.append(world_to_px((wx, wy)))
        color = (185, 91, 44) if obs.get("kind") == "wall" else (195, 119, 55)
        draw.polygon(corners, fill=color, outline=(107, 55, 35))
    else:
        cx, cy = world_to_px((obs["x"], obs["y"]))
        rr = int(float(obs["radius"]) / (2.0 * arena_half) * (size - 2 * margin))
        draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(190, 48, 42), outline=(105, 30, 28))
