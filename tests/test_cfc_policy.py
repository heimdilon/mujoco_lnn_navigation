import torch
from ncps.torch import CfC as NcpsCfC

from mujoco_lnn_nav.models.policies import CfCActorCritic, build_actor_critic


def test_cfc_policy_uses_ncps_recurrent_cfc():
    model = build_actor_critic("cfc", obs_dim=38, action_dim=2, hidden_size=16)

    assert isinstance(model, CfCActorCritic)
    assert isinstance(model.rnn, NcpsCfC)
    assert hasattr(model, "act_recurrent")
    assert hasattr(model, "forward_sequence")


def test_cfc_recurrent_shapes_and_state_update():
    torch.manual_seed(123)
    model = build_actor_critic("lnn", obs_dim=38, action_dim=2, hidden_size=16)
    obs = torch.randn(4, 38)
    state = model.initial_state(batch_size=4, device=torch.device("cpu"))

    action, log_prob, value, next_state = model.act_recurrent(obs, state, deterministic=True)

    assert state.shape == (4, 16)
    assert next_state.shape == (4, 16)
    assert action.shape == (4, 2)
    assert log_prob.shape == (4,)
    assert value.shape == (4,)
    assert not torch.allclose(state, next_state)


def test_cfc_sequence_forward_shapes():
    model = build_actor_critic("cfc", obs_dim=38, action_dim=2, hidden_size=16)
    obs_seq = torch.randn(7, 38)

    mean, value = model.forward_sequence(obs_seq)

    assert mean.shape == (7, 2)
    assert value.shape == (7,)
