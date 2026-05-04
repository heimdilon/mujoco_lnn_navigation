import numpy as np

from mujoco_lnn_nav.envs.layouts import ObstacleSpec
from mujoco_lnn_nav.envs.rays import cast_rays


def test_front_ray_hits_circle():
    obstacle = ObstacleSpec("cylinder", x=1.0, y=0.0, radius=0.2, half_x=0.2, half_y=0.2)
    rays = cast_rays(np.array([0.0, 0.0], dtype=np.float32), 0.0, [obstacle], 32, 5.0, 10.0)
    assert abs(float(rays[0]) - 0.8) < 0.05


def test_front_ray_does_not_treat_long_wall_as_circle():
    wall = ObstacleSpec("box", x=0.0, y=1.0, radius=3.0, half_x=3.0, half_y=0.08, yaw=0.0)
    rays = cast_rays(np.array([0.0, 0.0], dtype=np.float32), 0.0, [wall], 32, 5.0, 10.0)
    assert float(rays[0]) > 4.9
    assert abs(float(rays[8]) - 0.92) < 0.05
