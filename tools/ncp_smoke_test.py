"""End-to-end smoke test for NCP-wired CfC policy.

Loads the NCP train config YAML, builds the model through the standard factory,
exercises forward / recurrent / sequence APIs, and exports the wiring diagram —
the same path that real BC training will take.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mujoco_lnn_nav.models.policies import build_actor_critic


def main() -> None:
    cfg_path = ROOT / "configs" / "train" / "bc_ncp_cfc_dynamic_maps.yaml"
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    policy_name = cfg["policy"]
    hidden_size = int(cfg["hidden_size"])
    model_kwargs = dict(cfg.get("model_kwargs") or {})

    print(f"[1] Building {policy_name!r} via factory")
    print(f"    hidden_size={hidden_size}, model_kwargs={model_kwargs}")
    model = build_actor_critic(policy_name, 38, 2, hidden_size, **model_kwargs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"    type={type(model).__name__}, params={n_params}")
    print(f"    wiring: units={model.wiring.units}, output={model.ncp_output_size}")
    print(f"    wiring: synapses={model.wiring.synapse_count}, sensory={model.wiring.sensory_synapse_count}")
    print()

    print("[2] forward() single-step")
    obs = torch.randn(4, 38)
    mean, value = model(obs)
    print(f"    mean={mean.shape}, value={value.shape}")
    assert mean.shape == (4, 2) and value.shape == (4, 1)
    print()

    print("[3] act_recurrent() with state propagation")
    state = model.initial_state(4, torch.device("cpu"))
    print(f"    initial_state={state.shape}")
    action, log_prob, value, next_state = model.act_recurrent(obs, state, deterministic=True)
    print(f"    action={action.shape}, log_prob={log_prob.shape}, value={value.shape}, next_state={next_state.shape}")
    assert action.shape == (4, 2)
    assert next_state.shape == (4, model.ncp_units)
    assert not torch.allclose(state, next_state), "state must evolve after one step"
    print()

    print("[4] forward_sequence() — BC training path")
    obs_seq = torch.randn(2, 16, 38)
    mean_seq, value_seq = model.forward_sequence(obs_seq)
    print(f"    mean_seq={mean_seq.shape}, value_seq={value_seq.shape}")
    assert mean_seq.shape == (2, 16, 2)
    assert value_seq.shape == (2, 16)
    print()

    print("[5] BC-style loss backward() — gradient must flow")
    target = torch.randn_like(mean_seq)
    pred = torch.tanh(mean_seq)
    loss = (pred - target).pow(2).mean()
    loss.backward()
    encoder_grad = model.input[0].weight.grad
    assert encoder_grad is not None and torch.isfinite(encoder_grad).all()
    assert encoder_grad.abs().sum() > 0, "encoder gradient is zero — backprop is broken"
    print(f"    loss={loss.item():.5f}, encoder grad norm={encoder_grad.norm().item():.5f}")
    print()

    print("[6] Wiring diagram export")
    out_dir = ROOT / "report" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ncp_wiring_phase1_smoke.png"
    try:
        model.save_wiring_diagram(str(out_path))
        print(f"    saved: {out_path}")
    except Exception as exc:
        print(f"    WARN: draw_graph failed ({exc}) — non-fatal for training")
    print()

    print("ALL NCP SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
