# Dynamic LNN Navigation Start Note

Search date: 2026-05-11

## Local MuJoCo status

The MuJoCo environment now supports optional dynamic obstacles without changing the policy observation/action contract.

- Existing observation remains 38-D: goal distance, goal bearing, velocity, previous action, 32 LiDAR rays.
- Static map files remain compatible.
- Dynamic obstacles can be added under `map.dynamic_obstacles`.
- Supported motion types:
  - `line`: sinusoidal motion along an axis.
  - `circle`: circular motion around the obstacle's configured center.
- Evaluation JSON now records `obstacle_paths`; rollout PNG/GIF rendering uses those paths so moving obstacles are visible.

New starter maps:

- `configs/maps/dynamic_open_single.yaml`: easy open-field map with one moving cylinder.
- `configs/maps/dynamic_crossing.yaml`: harder corridor/crossing stress map with two moving obstacles.

Initial smoke results using `results/cfc_radius010_custom22_dagger2/latest.pt`:

| Map | Episodes | Success | Collision | Timeout | Mean steps | Mean final distance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dynamic_open_single` | 2 | 1.00 | 0.00 | 0.00 | 156.0 | 0.188 m |
| `dynamic_crossing` | 2 | 0.00 | 0.00 | 1.00 | 300.0 | 5.141 m |

Interpretation: the current CfC policy can handle a single moving obstacle in open space, but the corridor map exposes a goal-seeking/generalization weakness rather than immediate collision failure.

## Reproduction commands

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

```powershell
.\.venv\Scripts\python.exe scripts\batch_evaluate.py `
  --map-configs configs\maps\dynamic_open_single.yaml `
  --checkpoint results\cfc_radius010_custom22_dagger2\latest.pt `
  --episodes 2 --run-name dynamic_open_single_cfc_smoke `
  --max-steps 300 --device cpu
```

```powershell
.\.venv\Scripts\python.exe scripts\batch_evaluate.py `
  --map-configs configs\maps\dynamic_crossing.yaml `
  --checkpoint results\cfc_radius010_custom22_dagger2\latest.pt `
  --episodes 2 --run-name dynamic_crossing_cfc_smoke `
  --max-steps 300 --device cpu
```

## Literature scan

### Closest direct LNN navigation papers

1. Vorbach et al., "Causal Navigation by Continuous-time Neural Networks", NeurIPS 2021.
   - Link: https://papers.nips.cc/paper/2021/hash/67ba02d73c54f0b83c05507b7fb7267f-Abstract.html
   - Relevance: continuous-time/LTC-style models for visual drone navigation, including short/long-horizon navigation and chasing static/dynamic objects. Strong conceptual match for dynamic environments, but aerial and vision-based rather than ground/LiDAR.

2. Chahine et al., "Robust flight navigation out of distribution with liquid neural networks", Science Robotics 2023.
   - Link: https://www.science.org/doi/10.1126/scirobotics.adc8892
   - MIT summary: https://news.mit.edu/2023/drones-navigate-unseen-environments-liquid-neural-networks-0419
   - Relevance: LNN agents trained from demonstrations transfer across unseen environments and handle noise, occlusion, target rotation, and dynamic tracking. Again, aerial/vision rather than ground mobile robot.

3. Quach et al., "Gaussian Splatting to Real World Flight Navigation Transfer with Liquid Networks", CoRL/PMLR 2025.
   - Link: https://proceedings.mlr.press/v270/quach25a.html
   - Relevance: sim-to-real visual quadrotor navigation with Liquid Networks. Useful for our "simulation-to-dynamic-world" discussion, especially if we later compare MuJoCo/IsaacLab simulation transfer.

### Foundational LNN/CfC/NCP work

1. Hasani et al., "Liquid Time-constant Networks", AAAI 2021.
   - Link: https://ojs.aaai.org/index.php/AAAI/article/view/16936
   - Relevance: defines LTCs as continuous-time recurrent models with stable, bounded dynamics and strong sequence modeling.

2. Hasani et al., "Closed-form continuous-time neural networks", Nature Machine Intelligence 2022.
   - Link: https://www.nature.com/articles/s42256-022-00556-7
   - Relevance: CfC formulation used in this project; faster closed-form approximation of liquid dynamics, appropriate for real-time robot control.

3. Lechner et al., "Neural circuit policies enabling auditable autonomy", Nature Machine Intelligence 2020.
   - Link: https://www.nature.com/articles/s42256-020-00237-3
   - Relevance: compact LTC/NCP controller for autonomous vehicle lane keeping. Ground vehicle control is closer to our mobile-robot setting than the drone papers, but it is lane keeping, not 2-D obstacle navigation.

### Theses / dissertations

1. Ramin Hasani, "Interpretable recurrent neural networks in continuous-time control environments", PhD dissertation, TU Wien, 2020.
   - Link: https://repositum.tuwien.at/handle/20.500.12708/1068
   - Relevance: dissertation-level foundation for LTCs in continuous-time control, including robot and vehicle control.

2. Makram Chahine, "Dynamical Systems Perspectives on Efficient Physical Intelligence", MIT CSAIL thesis defense, May 11, 2026.
   - Link: https://www.csail.mit.edu/event/thesis-defense-makram-chahine-dynamical-systems-perspectives-efficient-physical-intelligence
   - Relevance: thesis abstract explicitly covers liquid neural networks for robust vision-based navigation under distribution shift and Gaussian-Splatting-based zero-shot sim-to-real transfer.

### Mobile robot navigation context

Al Mahmud et al., "Advancements and Challenges in Mobile Robot Navigation: A Comprehensive Review of Algorithms and Potential for Self-Learning Approaches", Journal of Intelligent & Robotic Systems 2024.

- Link: https://link.springer.com/article/10.1007/s10846-024-02149-5
- Relevance: broad mobile robot navigation survey. It positions self-learning methods as promising but data-hungry/risky and discusses LNNs as an emerging direction. This supports our framing that LNN/CfC for ground mobile robot navigation is still a research gap.

## Working conclusion

I did not find, in this first pass, a directly equivalent paper/thesis for "MuJoCo + differential-drive mobile robot + LiDAR + LNN/CfC navigation in dynamic maps." The closest prior work is:

- LNN/LTC/CfC for visual drone navigation and OOD robustness.
- NCP/LTC for autonomous ground vehicle lane keeping.
- General mobile robot navigation surveys that mention LNNs as a possible self-learning direction.

That gives the project a defensible niche: evaluate CfC/LNN policies for low-dimensional LiDAR mobile robot navigation under dynamic-obstacle map shifts, with GRU/LSTM/MLP baselines.

## Next experiment matrix

1. Static baseline re-check:
   - `custom_map_01`, `custom_map_02`, `custom_map_04`, `custom_map_05`, `custom_map_06`, `custom_map_07`
   - CfC, GRU, LSTM if compatible checkpoints exist.

2. Dynamic easy:
   - `dynamic_open_single`
   - 20 episodes, seeds fixed.

3. Dynamic stress:
   - `dynamic_crossing`
   - 20 episodes at 300/600/900 max steps.

4. Ablations:
   - obstacle speed/period sweep.
   - sensor noise sweep.
   - compare pure policy vs auto-waypoint-assisted diagnostic only.

5. Reporting metrics:
   - success, collision, timeout, mean steps, final distance.
   - min obstacle distance distribution.
   - dynamic obstacle speed/amplitude per run.
   - GIF/PNG rollout examples for success and failure.
