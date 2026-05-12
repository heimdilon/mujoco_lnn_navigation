import torch
from ncps.torch import CfC as NcpsCfC
from ncps.wirings import AutoNCP

from mujoco_lnn_nav.models.ncp_policies import NcpCfCActorCritic
from mujoco_lnn_nav.models.policies import build_actor_critic
from mujoco_lnn_nav.utils.checkpoints import load_policy_from_checkpoint


def _ncp_kwargs(units: int = 16, output_size: int = 4, sparsity: float = 0.5, seed: int = 22222) -> dict:
    return {
        "ncp_units": units,
        "ncp_output_size": output_size,
        "sparsity_level": sparsity,
        "wiring_seed": seed,
    }


def test_ncp_cfc_policy_uses_autoncp_wired_cfc():
    model = build_actor_critic("ncp_cfc", obs_dim=38, action_dim=2, hidden_size=8, **_ncp_kwargs())

    assert isinstance(model, NcpCfCActorCritic)
    assert isinstance(model.rnn, NcpsCfC)
    assert isinstance(model.wiring, AutoNCP)
    assert model.wiring.is_built()
    assert model.wiring.synapse_count > 0
    assert model.wiring.sensory_synapse_count > 0
    assert model.kind == "ncp_cfc"
    assert hasattr(model, "act_recurrent")
    assert hasattr(model, "forward_sequence")


def test_ncp_cfc_recurrent_shapes_and_state_update():
    torch.manual_seed(123)
    model = build_actor_critic("ncp_cfc", obs_dim=38, action_dim=2, hidden_size=8, **_ncp_kwargs(units=16, output_size=4))
    obs = torch.randn(4, 38)
    state = model.initial_state(batch_size=4, device=torch.device("cpu"))

    action, log_prob, value, next_state = model.act_recurrent(obs, state, deterministic=True)

    assert state.shape == (4, 16)
    assert next_state.shape == (4, 16)
    assert action.shape == (4, 2)
    assert log_prob.shape == (4,)
    assert value.shape == (4,)
    assert not torch.allclose(state, next_state)


def test_ncp_cfc_sequence_forward_shapes_unbatched():
    model = build_actor_critic("ncp_cfc", obs_dim=38, action_dim=2, hidden_size=8, **_ncp_kwargs())
    obs_seq = torch.randn(7, 38)

    mean, value = model.forward_sequence(obs_seq)

    assert mean.shape == (7, 2)
    assert value.shape == (7,)


def test_ncp_cfc_sequence_forward_shapes_batched():
    model = build_actor_critic("ncp_cfc", obs_dim=38, action_dim=2, hidden_size=8, **_ncp_kwargs())
    obs_seq = torch.randn(3, 7, 38)

    mean, value = model.forward_sequence(obs_seq)

    assert mean.shape == (3, 7, 2)
    assert value.shape == (3, 7)


def test_ncp_cfc_deterministic_wiring_with_seed():
    model_a = build_actor_critic("ncp_cfc", obs_dim=38, action_dim=2, hidden_size=8, **_ncp_kwargs(seed=42))
    model_b = build_actor_critic("ncp_cfc", obs_dim=38, action_dim=2, hidden_size=8, **_ncp_kwargs(seed=42))

    assert model_a.wiring.synapse_count == model_b.wiring.synapse_count
    assert model_a.wiring.sensory_synapse_count == model_b.wiring.sensory_synapse_count


def test_ncp_cfc_aliases_resolve_to_same_class():
    model_cfc = build_actor_critic("ncp_cfc", 38, 2, hidden_size=8, **_ncp_kwargs())
    model_alias = build_actor_critic("ncp_lnn", 38, 2, hidden_size=8, **_ncp_kwargs())
    model_short = build_actor_critic("ncp", 38, 2, hidden_size=8, **_ncp_kwargs())

    assert type(model_cfc) is NcpCfCActorCritic
    assert type(model_alias) is NcpCfCActorCritic
    assert type(model_short) is NcpCfCActorCritic


def test_ncp_cfc_bc_loss_backward_propagates():
    torch.manual_seed(7)
    model = build_actor_critic("ncp_cfc", obs_dim=38, action_dim=2, hidden_size=8, **_ncp_kwargs())
    obs_seq = torch.randn(1, 5, 38)
    target = torch.randn(1, 5, 2)

    mean, _ = model.forward_sequence(obs_seq)
    pred = torch.tanh(mean)
    loss = (pred - target).pow(2).mean()
    loss.backward()

    encoder_grad = model.input[0].weight.grad
    assert encoder_grad is not None
    assert torch.isfinite(encoder_grad).all()
    assert encoder_grad.abs().sum() > 0


def test_ncp_cfc_parameter_count_is_compact():
    """Sanity check: NCP-wired model is dramatically smaller than DeepCfC baseline."""
    model = build_actor_critic("ncp_cfc", obs_dim=38, action_dim=2, hidden_size=64, **_ncp_kwargs(units=48, output_size=16))
    deep = build_actor_critic("cfc_deep", obs_dim=38, action_dim=2, hidden_size=192)

    ncp_params = sum(p.numel() for p in model.parameters())
    deep_params = sum(p.numel() for p in deep.parameters())

    assert ncp_params * 10 < deep_params, (
        f"Expected NCP model to be at least 10x smaller, got NCP={ncp_params}, Deep={deep_params}"
    )


def test_ncp_cfc_checkpoint_loads_with_model_kwargs(tmp_path):
    kwargs = _ncp_kwargs(units=16, output_size=4, sparsity=0.5, seed=123)
    model = build_actor_critic("ncp_cfc", obs_dim=38, action_dim=2, hidden_size=8, **kwargs)
    checkpoint_path = tmp_path / "latest.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "policy": "ncp_cfc",
            "hidden_size": 8,
            "model_kwargs": kwargs,
        },
        checkpoint_path,
    )

    loaded = load_policy_from_checkpoint(checkpoint_path, obs_dim=38, action_dim=2)

    assert isinstance(loaded, NcpCfCActorCritic)
    assert loaded.ncp_units == 16
    assert loaded.ncp_output_size == 4
