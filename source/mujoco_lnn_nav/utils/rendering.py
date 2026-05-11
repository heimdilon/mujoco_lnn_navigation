from __future__ import annotations

from pathlib import Path
from math import atan2, cos, pi, sin

import numpy as np
from PIL import Image, ImageDraw

from mujoco_lnn_nav.envs.layouts import ObstacleSpec
from mujoco_lnn_nav.envs.rays import cast_rays


def _world_to_px_factory(config: dict, size: int, margin: int):
    arena_half = float(config["arena"]["half_size"])

    def world_to_px(point: tuple[float, float] | list[float]) -> tuple[int, int]:
        x = margin + (float(point[0]) + arena_half) / (2.0 * arena_half) * (size - 2 * margin)
        y = size - margin - (float(point[1]) + arena_half) / (2.0 * arena_half) * (size - 2 * margin)
        return int(round(x)), int(round(y))

    return world_to_px


def _obstacle_dicts_at(episode: dict, frame_index: int | None = None) -> list[dict]:
    obstacles = [dict(obs) for obs in episode.get("obstacles", [])]
    paths = episode.get("obstacle_paths", [])
    for idx, path in enumerate(paths):
        if idx >= len(obstacles) or not path:
            continue
        source_index = len(path) - 1 if frame_index is None else min(frame_index, len(path) - 1)
        position = path[source_index]
        obstacles[idx]["x"] = float(position[0])
        obstacles[idx]["y"] = float(position[1])
    return obstacles


def _draw_static_map(draw: ImageDraw.ImageDraw, config: dict, episode: dict, size: int, margin: int, frame_index: int | None = None) -> None:
    arena_half = float(config["arena"]["half_size"])
    world_to_px = _world_to_px_factory(config, size, margin)
    draw.rectangle([margin, margin, size - margin, size - margin], outline=(70, 76, 82), width=2)
    for path in episode.get("obstacle_paths", []):
        if len(path) > 1:
            last_index = len(path) - 1 if frame_index is None else min(frame_index, len(path) - 1)
            trail = [world_to_px(point) for point in path[: last_index + 1]]
            if len(trail) > 1:
                draw.line(trail, fill=(116, 119, 124), width=2)
    for obs in _obstacle_dicts_at(episode, frame_index):
        cx, cy = world_to_px((obs["x"], obs["y"]))
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
            draw.polygon(corners, fill=(193, 92, 45), outline=(107, 55, 35))
        else:
            rr = int(obs["radius"] / (2.0 * arena_half) * (size - 2 * margin))
            draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(190, 48, 42), outline=(105, 30, 28))

    goal = episode.get("goal", [0.0, 0.0])
    gx, gy = world_to_px(goal)
    draw.ellipse([gx - 11, gy - 11, gx + 11, gy + 11], fill=(39, 155, 78), outline=(20, 90, 45), width=2)
    draw.ellipse([gx - 4, gy - 4, gx + 4, gy + 4], fill=(245, 247, 244))
    for waypoint in episode.get("waypoints", [])[:-1]:
        wx, wy = world_to_px(waypoint)
        draw.ellipse([wx - 5, wy - 5, wx + 5, wy + 5], fill=(88, 129, 220), outline=(39, 75, 160))


def _episode_obstacles(episode: dict, frame_index: int | None = None) -> list[ObstacleSpec]:
    obstacles = []
    for obs in _obstacle_dicts_at(episode, frame_index):
        obstacles.append(
            ObstacleSpec(
                str(obs["shape"]),
                float(obs["x"]),
                float(obs["y"]),
                float(obs["radius"]),
                float(obs["half_x"]),
                float(obs["half_y"]),
                float(obs.get("yaw", 0.0)),
            )
        )
    return obstacles


def _yaw_at(episode: dict, index: int) -> float:
    yaws = episode.get("yaws", [])
    if yaws:
        return float(yaws[min(index, len(yaws) - 1)])
    path = episode.get("path", [])
    if len(path) < 2:
        return 0.0
    current = path[min(index, len(path) - 1)]
    other = path[min(index + 1, len(path) - 1)] if index < len(path) - 1 else path[index - 1]
    return float(atan2(float(other[1]) - float(current[1]), float(other[0]) - float(current[0])))


def _draw_lidar(
    image: Image.Image,
    config: dict,
    episode: dict,
    point: list[float] | tuple[float, float],
    yaw: float,
    size: int,
    margin: int,
    frame_index: int | None = None,
) -> Image.Image:
    num_rays = int(config["sensors"]["rays"])
    max_range = float(config["sensors"]["max_range"])
    arena_half = float(config["arena"]["half_size"])
    ranges = cast_rays(
        origin=np.array([float(point[0]), float(point[1])], dtype=np.float32),
        yaw=yaw,
        obstacles=_episode_obstacles(episode, frame_index),
        num_rays=num_rays,
        max_range=max_range,
        arena_half=arena_half,
    )
    world_to_px = _world_to_px_factory(config, size, margin)
    origin_px = world_to_px(point)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for ray_index, ray_distance in enumerate(ranges):
        angle = yaw + 2.0 * pi * ray_index / float(num_rays)
        end = (
            float(point[0]) + cos(angle) * float(ray_distance),
            float(point[1]) + sin(angle) * float(ray_distance),
        )
        end_px = world_to_px(end)
        alpha = 90 if ray_index % 4 == 0 else 52
        draw.line([origin_px, end_px], fill=(16, 142, 180, alpha), width=1)
        if float(ray_distance) < max_range * 0.98:
            draw.ellipse([end_px[0] - 2, end_px[1] - 2, end_px[0] + 2, end_px[1] + 2], fill=(8, 115, 150, 115))
    return Image.alpha_composite(image.convert("RGBA"), overlay)


def render_rollout_png(config: dict, episode: dict, path: Path, size: int = 900) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    margin = 42
    image = Image.new("RGBA", (size, size), (246, 247, 244, 255))
    draw = ImageDraw.Draw(image)
    world_to_px = _world_to_px_factory(config, size, margin)

    raw_path = episode.get("path", [])
    final_index = len(raw_path) - 1 if raw_path else None
    _draw_static_map(draw, config, episode, size, margin, final_index)
    if raw_path:
        image = _draw_lidar(image, config, episode, raw_path[-1], _yaw_at(episode, len(raw_path) - 1), size, margin, final_index)
        draw = ImageDraw.Draw(image)

    path_points = [world_to_px(point) for point in episode.get("path", [])]
    if len(path_points) > 1:
        draw.line(path_points, fill=(25, 82, 170), width=4, joint="curve")
    if path_points:
        sx, sy = path_points[0]
        ex, ey = path_points[-1]
        draw.ellipse([sx - 7, sy - 7, sx + 7, sy + 7], fill=(250, 198, 64), outline=(120, 90, 20))
        draw.ellipse([ex - 7, ey - 7, ex + 7, ey + 7], fill=(24, 92, 185), outline=(12, 40, 95))

    distances = episode.get("distances", [])
    if len(distances) >= 2:
        text = f"distance {distances[0]:.2f} -> {distances[-1]:.2f} m"
    else:
        text = "rollout"
    draw.text((margin, 14), text, fill=(35, 39, 44))
    image.convert("RGB").save(path)


def render_rollout_gif(
    config: dict,
    episode: dict,
    path: Path,
    size: int = 720,
    duration_ms: int = 45,
    max_frames: int = 120,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    margin = 34
    world_to_px = _world_to_px_factory(config, size, margin)
    raw_path = episode.get("path", [])
    if not raw_path:
        return

    stride = max(1, len(raw_path) // max_frames)
    points = raw_path[::stride]
    if points[-1] != raw_path[-1]:
        points.append(raw_path[-1])

    frames: list[Image.Image] = []
    for idx in range(len(points)):
        image = Image.new("RGBA", (size, size), (246, 247, 244, 255))
        draw = ImageDraw.Draw(image)
        source_index = min(idx * stride, len(raw_path) - 1)
        _draw_static_map(draw, config, episode, size, margin, source_index)
        image = _draw_lidar(image, config, episode, points[idx], _yaw_at(episode, source_index), size, margin, source_index)
        draw = ImageDraw.Draw(image)
        trail = [world_to_px(point) for point in points[: idx + 1]]
        if len(trail) > 1:
            draw.line(trail, fill=(25, 82, 170), width=4, joint="curve")
        x, y = trail[-1]
        draw.ellipse([x - 9, y - 9, x + 9, y + 9], fill=(24, 92, 185), outline=(12, 40, 95), width=2)
        distances = episode.get("distances", [])
        if distances:
            distance_idx = min(idx * stride, len(distances) - 1)
            text = f"step {min(idx * stride, len(raw_path) - 1):03d}  distance {distances[distance_idx]:.2f} m"
        else:
            text = f"step {min(idx * stride, len(raw_path) - 1):03d}"
        draw.text((margin, 10), text, fill=(35, 39, 44))
        frames.append(image.convert("RGB"))

    frames.extend([frames[-1].copy() for _ in range(10)])
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, optimize=True)
