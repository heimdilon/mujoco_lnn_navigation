from __future__ import annotations

from copy import deepcopy
from heapq import heappop, heappush
from math import hypot
from typing import Any

import numpy as np

from mujoco_lnn_nav.envs.layouts import box_signed_distance, fixed_obstacles, is_free


def _clearance(point: np.ndarray, obstacles: list) -> float:
    if not obstacles:
        return 10.0
    best = 10.0
    for obs in obstacles:
        if obs.shape == "box":
            best = min(best, box_signed_distance(point, obs))
        else:
            best = min(best, float(np.linalg.norm(point - obs.center) - obs.radius))
    return best


def _to_grid(point: tuple[float, float] | list[float], arena_half: float, resolution: float) -> tuple[int, int]:
    return (
        int(round((float(point[0]) + arena_half) / resolution)),
        int(round((float(point[1]) + arena_half) / resolution)),
    )


def _to_world(cell: tuple[int, int], arena_half: float, resolution: float) -> tuple[float, float]:
    return (cell[0] * resolution - arena_half, cell[1] * resolution - arena_half)


def _line_is_free(a: tuple[float, float], b: tuple[float, float], cfg: dict[str, Any], samples: int = 18) -> bool:
    obstacles = fixed_obstacles(cfg)
    for idx in range(samples + 1):
        t = idx / samples
        point = np.array([a[0] * (1.0 - t) + b[0] * t, a[1] * (1.0 - t) + b[1] * t], dtype=np.float32)
        if not is_free(point, obstacles, cfg, padding=0.04):
            return False
    return True


def _nearest_free(cell: tuple[int, int], free: np.ndarray) -> tuple[int, int]:
    if free[cell[1], cell[0]]:
        return cell
    height, width = free.shape
    for radius in range(1, max(width, height)):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx = cell[0] + dx
                ny = cell[1] + dy
                if 0 <= nx < width and 0 <= ny < height and free[ny, nx]:
                    return nx, ny
    raise RuntimeError("No free cell found in manual map.")


def plan_waypoints(cfg: dict[str, Any], resolution: float = 0.12, simplify: bool = True) -> list[list[float]]:
    map_cfg = cfg.get("map", {})
    start = map_cfg["start"]
    goal = map_cfg["goal"]
    arena_half = float(cfg["arena"]["half_size"])
    width = int(round(2.0 * arena_half / resolution)) + 1
    height = width
    free = np.zeros((height, width), dtype=bool)
    clearance = np.zeros((height, width), dtype=np.float32)
    obstacles = fixed_obstacles(cfg)
    for y in range(height):
        for x in range(width):
            point = np.array(_to_world((x, y), arena_half, resolution), dtype=np.float32)
            free[y, x] = is_free(point, obstacles, cfg, padding=0.05)
            clearance[y, x] = _clearance(point, obstacles)

    start_cell = _nearest_free(_to_grid(start[:2], arena_half, resolution), free)
    goal_cell = _nearest_free(_to_grid(goal[:2], arena_half, resolution), free)
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    cost_so_far = {start_cell: 0.0}
    queue: list[tuple[float, tuple[int, int]]] = []
    heappush(queue, (0.0, start_cell))
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while queue:
        _, current = heappop(queue)
        if current == goal_cell:
            break
        for dx, dy in neighbors:
            nxt = (current[0] + dx, current[1] + dy)
            if not (0 <= nxt[0] < width and 0 <= nxt[1] < height) or not free[nxt[1], nxt[0]]:
                continue
            cell_clearance = float(clearance[nxt[1], nxt[0]])
            clearance_penalty = max(0.0, 0.55 - cell_clearance) * 4.0
            step_cost = hypot(dx, dy) * (1.0 + clearance_penalty)
            new_cost = cost_so_far[current] + step_cost
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + hypot(goal_cell[0] - nxt[0], goal_cell[1] - nxt[1])
                came_from[nxt] = current
                heappush(queue, (priority, nxt))

    if goal_cell not in came_from and goal_cell != start_cell:
        raise RuntimeError(f"No A* path found for map {cfg.get('name', 'manual_map')}.")

    cells = [goal_cell]
    while cells[-1] != start_cell:
        cells.append(came_from[cells[-1]])
    cells.reverse()
    raw = [_to_world(cell, arena_half, resolution) for cell in cells]
    raw[0] = (float(start[0]), float(start[1]))
    raw[-1] = (float(goal[0]), float(goal[1]))

    if not simplify:
        dense = [raw[idx] for idx in range(4, len(raw), 4)]
        if dense[-1] != raw[-1]:
            dense.append(raw[-1])
        return [[float(x), float(y)] for x, y in dense]

    simplified = [raw[0]]
    anchor = 0
    while anchor < len(raw) - 1:
        best = anchor + 1
        for candidate in range(len(raw) - 1, anchor, -1):
            if _line_is_free(raw[anchor], raw[candidate], cfg):
                best = candidate
                break
        simplified.append(raw[best])
        anchor = best
    return [[float(x), float(y)] for x, y in simplified[1:]]


def with_auto_waypoints(
    cfg: dict[str, Any],
    resolution: float = 0.12,
    waypoint_radius: float = 0.45,
    simplify: bool = True,
) -> dict[str, Any]:
    updated = deepcopy(cfg)
    updated.setdefault("map", {})["waypoints"] = plan_waypoints(updated, resolution=resolution, simplify=simplify)
    updated["map"]["waypoint_radius"] = float(waypoint_radius)
    return updated
