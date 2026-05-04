import mujoco

from mujoco_lnn_nav.config import load_task_config
from mujoco_lnn_nav.envs.layouts import max_obstacles
from mujoco_lnn_nav.envs.xml import build_navigation_xml


def test_xml_compiles():
    cfg = load_task_config("configs/task/sparse_goal.yaml")
    model = mujoco.MjModel.from_xml_string(build_navigation_xml(cfg, max_obstacles(cfg)))
    assert model.nq >= 3
    assert model.nu == 3

