from __future__ import annotations

from math import cos, sin

import numpy as np

from mujoco_lnn_nav.envs.layouts import ObstacleSpec


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _ray_circle(origin: np.ndarray, direction: np.ndarray, center: np.ndarray, radius: float) -> float | None:
    oc = origin - center
    b = 2.0 * float(np.dot(oc, direction))
    c = float(np.dot(oc, oc) - radius * radius)
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return None
    root = float(np.sqrt(disc))
    t0 = (-b - root) / 2.0
    t1 = (-b + root) / 2.0
    hits = [t for t in (t0, t1) if t >= 0.0]
    return min(hits) if hits else None


def _ray_box(origin: np.ndarray, direction: np.ndarray, obs: ObstacleSpec) -> float | None:
    dx = float(origin[0]) - obs.x
    dy = float(origin[1]) - obs.y
    c = cos(obs.yaw)
    s = sin(obs.yaw)
    local_origin = np.array([c * dx + s * dy, -s * dx + c * dy], dtype=np.float32)
    local_direction = np.array(
        [c * float(direction[0]) + s * float(direction[1]), -s * float(direction[0]) + c * float(direction[1])],
        dtype=np.float32,
    )
    lower = np.array([-obs.half_x, -obs.half_y], dtype=np.float32)
    upper = np.array([obs.half_x, obs.half_y], dtype=np.float32)
    tmin = -np.inf
    tmax = np.inf
    for axis in range(2):
        if abs(float(local_direction[axis])) < 1e-8:
            if local_origin[axis] < lower[axis] or local_origin[axis] > upper[axis]:
                return None
            continue
        inv_d = 1.0 / float(local_direction[axis])
        t1 = float((lower[axis] - local_origin[axis]) * inv_d)
        t2 = float((upper[axis] - local_origin[axis]) * inv_d)
        tmin = max(tmin, min(t1, t2))
        tmax = min(tmax, max(t1, t2))
    if tmax < 0.0 or tmin > tmax:
        return None
    return max(tmin, 0.0)


def _ray_box_batch(origin: np.ndarray, directions: np.ndarray, obs: ObstacleSpec, max_range: float) -> np.ndarray:
    dx = float(origin[0]) - obs.x
    dy = float(origin[1]) - obs.y
    c = cos(obs.yaw)
    s = sin(obs.yaw)
    local_origin = np.array([c * dx + s * dy, -s * dx + c * dy], dtype=np.float32)
    local_directions = np.stack(
        [
            c * directions[:, 0] + s * directions[:, 1],
            -s * directions[:, 0] + c * directions[:, 1],
        ],
        axis=1,
    ).astype(np.float32)
    lower = np.array([-obs.half_x, -obs.half_y], dtype=np.float32)
    upper = np.array([obs.half_x, obs.half_y], dtype=np.float32)
    tmin = np.full((directions.shape[0],), -np.inf, dtype=np.float32)
    tmax = np.full((directions.shape[0],), np.inf, dtype=np.float32)
    valid = np.ones((directions.shape[0],), dtype=bool)

    for axis in range(2):
        axis_direction = local_directions[:, axis]
        parallel = np.abs(axis_direction) < 1e-8
        outside = (local_origin[axis] < lower[axis]) | (local_origin[axis] > upper[axis])
        valid &= ~(parallel & outside)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_d = 1.0 / axis_direction
            t1 = (lower[axis] - local_origin[axis]) * inv_d
            t2 = (upper[axis] - local_origin[axis]) * inv_d
        axis_min = np.minimum(t1, t2)
        axis_max = np.maximum(t1, t2)
        axis_min = np.where(parallel, -np.inf, axis_min)
        axis_max = np.where(parallel, np.inf, axis_max)
        tmin = np.maximum(tmin, axis_min.astype(np.float32))
        tmax = np.minimum(tmax, axis_max.astype(np.float32))

    valid &= (tmax >= 0.0) & (tmin <= tmax)
    return np.where(valid, np.maximum(tmin, 0.0), max_range).astype(np.float32)


def cast_rays(
    origin: np.ndarray,
    yaw: float,
    obstacles: list[ObstacleSpec],
    num_rays: int,
    max_range: float,
    arena_half: float,
) -> np.ndarray:
    return cast_rays_fast(origin, yaw, obstacles, num_rays, max_range, arena_half)


def cast_rays_exact(
    origin: np.ndarray,
    yaw: float,
    obstacles: list[ObstacleSpec],
    num_rays: int,
    max_range: float,
    arena_half: float,
) -> np.ndarray:
    ranges = np.full((num_rays,), max_range, dtype=np.float32)
    for idx in range(num_rays):
        angle = yaw + (2.0 * np.pi * idx / num_rays)
        direction = np.array([cos(angle), sin(angle)], dtype=np.float32)
        for obs in obstacles:
            if obs.shape == "box":
                hit = _ray_box(origin, direction, obs)
            else:
                hit = _ray_circle(origin, direction, obs.center, obs.radius)
            if hit is not None and hit < ranges[idx]:
                ranges[idx] = max(0.0, float(hit))

        for axis in range(2):
            if abs(float(direction[axis])) < 1e-8:
                continue
            boundary = arena_half if direction[axis] > 0.0 else -arena_half
            t = (boundary - origin[axis]) / direction[axis]
            if t >= 0.0:
                ranges[idx] = min(ranges[idx], float(t))
    return np.clip(ranges, 0.0, max_range)


def cast_rays_fast(
    origin: np.ndarray,
    yaw: float,
    obstacles: list[ObstacleSpec],
    num_rays: int,
    max_range: float,
    arena_half: float,
) -> np.ndarray:
    angles = yaw + 2.0 * np.pi * np.arange(num_rays, dtype=np.float32) / float(num_rays)
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    ranges = np.full((num_rays,), max_range, dtype=np.float32)

    for obs in obstacles:
        if obs.shape == "box":
            ranges = np.minimum(ranges, _ray_box_batch(origin, directions, obs, max_range))
            continue
        center = obs.center.astype(np.float32)
        radius = obs.radius
        oc = origin.astype(np.float32) - center
        b = 2.0 * (directions @ oc)
        c = float(oc @ oc - radius * radius)
        disc = b * b - 4.0 * c
        valid = disc >= 0.0
        if not bool(np.any(valid)):
            continue
        root = np.sqrt(np.maximum(disc, 0.0))
        t0 = (-b - root) * 0.5
        t1 = (-b + root) * 0.5
        hit = np.where(t0 >= 0.0, t0, np.where(t1 >= 0.0, t1, max_range))
        hit = np.where(valid, hit, max_range)
        ranges = np.minimum(ranges, hit.astype(np.float32))

    for axis in range(2):
        direction_axis = directions[:, axis]
        boundary = np.where(direction_axis >= 0.0, arena_half, -arena_half).astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (boundary - float(origin[axis])) / direction_axis
        t = np.where((np.abs(direction_axis) > 1e-8) & (t >= 0.0), t, max_range)
        ranges = np.minimum(ranges, t.astype(np.float32))
    return np.clip(ranges, 0.0, max_range).astype(np.float32)
