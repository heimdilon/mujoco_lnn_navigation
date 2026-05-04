# MuJoCo LNN Navigation

Clean MuJoCo project for comparing CfC/LNN-style, MLP, GRU, and LSTM policies on the same differential-drive navigation contract.

This project is intentionally separate from the previous Isaac work. It does not import Isaac, IsaacLab, or kit runtime modules.

## Contract

- Observation dimension: `38`
- Action dimension: `2`
- Observation layout:
  - normalized goal distance
  - normalized goal bearing
  - normalized linear velocity
  - normalized angular velocity
  - previous action, two values
  - 32 normalized ray ranges
- Action layout:
  - `[linear, angular]`
  - normalized to `[-1, 1]`

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest
```

Run a short smoke training job:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --task-config configs\task\sparse_goal.yaml --train-config configs\train\ppo_mlp.yaml --run-name sparse_goal_smoke --steps 4096 --num-envs 8 --eval-episodes 8
```

Evaluate a checkpoint and write JSON/CSV/PNG output:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --task-config configs\task\sparse_goal.yaml --checkpoint results\sparse_goal_smoke\latest.pt --episodes 32 --run-name sparse_goal_eval
```

Evaluate one checkpoint across multiple maps and write a summary table:

```powershell
.\.venv\Scripts\python.exe scripts\batch_evaluate.py --map-configs configs\maps\custom_map_01.yaml configs\maps\custom_map_02.yaml --checkpoint results\custom_maps_gru_bc_dagger\latest.pt --episodes 4 --run-name batch_manual_maps_pure_gru_dagger --max-steps 900 --goal-observation-max 10
```

This writes `summary.csv`, `summary.json`, and per-map `eval.csv`, `eval.json`, `rollout.png`, and `rollout.gif` under `results/<run-name>`. Leave `--auto-waypoints` off for pure policy evaluation.

Reproduce the first midterm result with the trained GRU + BC/DAgger checkpoint:

```powershell
.\.venv\Scripts\python.exe scripts\batch_evaluate.py --map-configs configs\maps\custom_map_01.yaml configs\maps\custom_map_02.yaml --checkpoint results\custom_maps_gru_bc_dagger\latest.pt --episodes 4 --run-name batch_manual_maps_pure_gru_dagger --max-steps 900 --goal-observation-max 10
```

The resulting evaluation is pure policy evaluation: do not pass `--auto-waypoints`.

Run a quick policy comparison across MLP, CfC/LNN, GRU, and LSTM:

```powershell
.\.venv\Scripts\python.exe scripts\compare_policies.py --map-configs configs\maps\custom_map_01.yaml configs\maps\custom_map_02.yaml --run-name policy_comparison_quick_manual --policies mlp cfc gru lstm --epochs 60 --dagger-iterations 1 --dagger-rollouts-per-map 2 --dagger-epochs 15 --episodes 4 --max-steps 900 --goal-observation-max 10 --no-gif --no-png
```

This trains each policy with the same behavioral-cloning plus DAgger pipeline, evaluates with A* off, and writes `comparison_summary.csv`, `policy_summary.csv`, and `comparison_summary.json`.

Optional MuJoCo viewer:

```powershell
.\.venv\Scripts\python.exe scripts\watch_mujoco.py --task-config configs\task\sparse_goal.yaml --checkpoint results\sparse_goal_smoke\latest.pt
```

Open the map editor:

```powershell
.\.venv\Scripts\python.exe scripts\map_editor.py --port 8765
```

Build the midterm LaTeX report:

```powershell
xelatex -interaction=nonstopmode -halt-on-error -output-directory=report\build report\main.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=report\build report\main.tex
```

The PDF is written to `report\build\main.pdf`.

The editor writes fixed-map task configs under `configs/maps`. Use them directly:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --task-config configs\maps\custom_map_01.yaml --checkpoint results\open_clutter_100k_gru\latest.pt --episodes 8 --run-name custom_map_eval
```

## Acceptance Gates

- `sparse_goal`, 32 episode eval: success rate at least `0.70`
- `open_clutter`, 32 episode eval: success rate at least `0.40`
- Evaluation must emit a top-down rollout PNG.
- Source should contain no Isaac, IsaacLab, or `kit.exe` dependency.
