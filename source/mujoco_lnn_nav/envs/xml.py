from __future__ import annotations


def build_navigation_xml(config: dict, max_obstacles: int) -> str:
    arena_half = float(config["arena"]["half_size"])
    robot_radius = float(config["robot"]["radius"])
    max_linear = float(config["robot"]["max_linear_velocity"])
    max_angular = float(config["robot"]["max_angular_velocity"])
    timestep = float(config.get("physics_dt", 0.02))

    obstacle_geoms: list[str] = []
    for idx in range(max_obstacles):
        obstacle_geoms.append(
            f"""
      <geom name="obstacle_cyl_{idx}" type="cylinder" pos="100 100 -10" size="0.1 0.35"
            rgba="0.76 0.18 0.15 0" contype="0" conaffinity="0"/>
      <geom name="obstacle_box_{idx}" type="box" pos="100 100 -10" size="0.1 0.1 0.35"
            rgba="0.76 0.31 0.12 0" contype="0" conaffinity="0"/>
"""
        )

    return f"""
<mujoco model="mujoco_lnn_navigation">
  <compiler angle="radian"/>
  <option timestep="{timestep}" gravity="0 0 -9.81" integrator="Euler"/>

  <default>
    <geom condim="3" friction="1.0 0.02 0.002"/>
  </default>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.88 0.88 0.86" rgb2="0.78 0.80 0.78"
             width="256" height="256"/>
    <material name="floor_mat" texture="grid" texrepeat="8 8" reflectance="0.05"/>
  </asset>

  <worldbody>
    <light name="key" pos="0 -3 5" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="{arena_half} {arena_half} 0.1" material="floor_mat"/>
    <body name="goal" mocap="true" pos="0 0 0.04">
      <geom name="goal_marker" type="sphere" size="0.11" rgba="0.10 0.65 0.22 0.75"
            contype="0" conaffinity="0"/>
    </body>
    <body name="robot" pos="0 0 0.08">
      <joint name="slide_x" type="slide" axis="1 0 0" damping="0"/>
      <joint name="slide_y" type="slide" axis="0 1 0" damping="0"/>
      <joint name="yaw" type="hinge" axis="0 0 1" damping="0"/>
      <geom name="base" type="cylinder" size="{robot_radius} 0.055" rgba="0.12 0.32 0.74 1"/>
      <geom name="nose" type="box" pos="{robot_radius * 0.78} 0 0.035"
            size="0.04 0.025 0.02" rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0"/>
      <geom name="left_wheel" type="cylinder" pos="0 0.18 0" euler="1.5708 0 0"
            size="0.055 0.018" rgba="0.02 0.02 0.02 1" contype="0" conaffinity="0"/>
      <geom name="right_wheel" type="cylinder" pos="0 -0.18 0" euler="1.5708 0 0"
            size="0.055 0.018" rgba="0.02 0.02 0.02 1" contype="0" conaffinity="0"/>
    </body>
    {''.join(obstacle_geoms)}
  </worldbody>

  <actuator>
    <velocity name="vx_motor" joint="slide_x" kv="80" ctrlrange="-{max_linear} {max_linear}"/>
    <velocity name="vy_motor" joint="slide_y" kv="80" ctrlrange="-{max_linear} {max_linear}"/>
    <velocity name="yaw_motor" joint="yaw" kv="45" ctrlrange="-{max_angular} {max_angular}"/>
  </actuator>
</mujoco>
"""

