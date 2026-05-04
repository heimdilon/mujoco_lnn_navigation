from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mujoco_lnn_nav.config import load_task_config
from mujoco_lnn_nav.envs.navigation import MujocoNavigationEnv
from mujoco_lnn_nav.utils.checkpoints import load_policy_from_checkpoint
from mujoco_lnn_nav.utils.evaluation import heuristic_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_cfg = load_task_config(args.task_config)
    device = torch.device(args.device)
    env = MujocoNavigationEnv(task_cfg, num_envs=1, device=device, auto_reset=True)
    model = None
    if args.checkpoint:
        model = load_policy_from_checkpoint(args.checkpoint, env.observation_dim, env.action_dim, device)
    obs = env.reset()
    single = env.envs[0]
    with mujoco.viewer.launch_passive(single.model, single.data) as viewer:
        while viewer.is_running():
            if model is None:
                action = heuristic_action(env)
            else:
                with torch.no_grad():
                    action, _, _ = model.act(obs, deterministic=True)
            out = env.step(action)
            obs = out.observation
            mujoco.mj_forward(single.model, single.data)
            viewer.sync()
            time.sleep(float(task_cfg.get("physics_dt", 0.02)) * int(task_cfg.get("frame_skip", 4)))


if __name__ == "__main__":
    main()

