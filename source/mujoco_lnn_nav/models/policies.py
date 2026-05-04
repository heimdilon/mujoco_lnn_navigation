from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal
from ncps.torch import CfC as NcpsCfC


def _mlp(input_dim: int, hidden_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size),
        nn.Tanh(),
    )


class BaseActorCritic(nn.Module):
    def _dist_value(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        mean, value = self.forward(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std), value.squeeze(-1)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self._dist_value(obs)
        raw_action = dist.mean if deterministic else dist.rsample()
        action = torch.tanh(raw_action)
        correction = torch.log(1.0 - action.pow(2) + 1e-6).sum(-1)
        log_prob = dist.log_prob(raw_action).sum(-1) - correction
        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eps = 1e-6
        clipped = torch.clamp(action, -1.0 + eps, 1.0 - eps)
        raw_action = 0.5 * (torch.log1p(clipped) - torch.log1p(-clipped))
        dist, value = self._dist_value(obs)
        log_prob = dist.log_prob(raw_action).sum(-1)
        correction = torch.log(1.0 - clipped.pow(2) + eps).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob - correction, entropy, value


class MlpActorCritic(BaseActorCritic):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()
        self.policy_impl = "mlp"
        self.encoder = _mlp(obs_dim, hidden_size)
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(obs)
        return self.actor(features), self.critic(features)


def _squashed_normal_action(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    std = torch.exp(log_std).expand_as(mean)
    dist = Normal(mean, std)
    raw_action = mean if deterministic else dist.rsample()
    action = torch.tanh(raw_action)
    correction = torch.log(1.0 - action.pow(2) + 1e-6).sum(-1)
    log_prob = dist.log_prob(raw_action).sum(-1) - correction
    return action, log_prob


class CfCActorCritic(BaseActorCritic):
    """Recurrent CfC/LNN policy backed by `ncps.torch.CfC`."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()
        self.kind = "cfc"
        self.policy_impl = "ncps.torch.CfC"
        self.rnn = NcpsCfC(
            obs_dim,
            hidden_size,
            return_sequences=True,
            batch_first=True,
            mode="default",
            backbone_units=hidden_size,
            backbone_layers=1,
            backbone_dropout=0.0,
        )
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output, _ = self.rnn(obs.unsqueeze(1))
        features = output[:, -1]
        return self.actor(features), self.critic(features)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((batch_size, self.actor.in_features), dtype=torch.float32, device=device)

    def reset_state_indices(self, state: torch.Tensor, indices: list[int]) -> torch.Tensor:
        if indices:
            state[indices, :] = 0.0
        return state

    def forward_recurrent(self, obs: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output, next_state = self.rnn(obs.unsqueeze(1), state)
        features = output[:, -1]
        return self.actor(features), self.critic(features), next_state

    def act_recurrent(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, value, next_state = self.forward_recurrent(obs, state)
        action, log_prob = _squashed_normal_action(mean, self.log_std, deterministic=deterministic)
        return action, log_prob, value.squeeze(-1), next_state

    def forward_sequence(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        is_unbatched = obs.dim() == 2
        sequence = obs.unsqueeze(0) if is_unbatched else obs
        output, _ = self.rnn(sequence)
        mean = self.actor(output)
        value = self.critic(output).squeeze(-1)
        if is_unbatched:
            mean = mean.squeeze(0)
            value = value.squeeze(0)
        return mean, value


class RecurrentActorCritic(BaseActorCritic):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128, kind: str = "gru"):
        super().__init__()
        self.kind = kind
        self.policy_impl = f"torch.nn.{kind.upper()}"
        self.input = nn.Sequential(nn.Linear(obs_dim, hidden_size), nn.Tanh())
        if kind == "gru":
            self.rnn: nn.Module = nn.GRU(hidden_size, hidden_size, batch_first=True)
        elif kind == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        else:
            raise ValueError(f"Unknown recurrent policy kind: {kind}")
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.input(obs).unsqueeze(1)
        output, _ = self.rnn(features)
        features = output[:, -1]
        return self.actor(features), self.critic(features)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden_size = self.actor.in_features
        h = torch.zeros((1, batch_size, hidden_size), dtype=torch.float32, device=device)
        if self.kind == "lstm":
            return h, torch.zeros_like(h)
        return h

    def reset_state_indices(
        self,
        state: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        indices: list[int],
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not indices:
            return state
        if isinstance(state, tuple):
            h, c = state
            h[:, indices, :] = 0.0
            c[:, indices, :] = 0.0
            return h, c
        state[:, indices, :] = 0.0
        return state

    def forward_recurrent(
        self,
        obs: torch.Tensor,
        state: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        features = self.input(obs).unsqueeze(1)
        output, next_state = self.rnn(features, state)
        features = output[:, -1]
        return self.actor(features), self.critic(features), next_state

    def act_recurrent(
        self,
        obs: torch.Tensor,
        state: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        mean, value, next_state = self.forward_recurrent(obs, state)
        action, log_prob = _squashed_normal_action(mean, self.log_std, deterministic=deterministic)
        return action, log_prob, value.squeeze(-1), next_state

    def forward_sequence(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.input(obs)
        output, _ = self.rnn(features)
        return self.actor(output), self.critic(output).squeeze(-1)


def build_actor_critic(policy: str, obs_dim: int, action_dim: int, hidden_size: int = 128) -> BaseActorCritic:
    policy = policy.lower()
    if policy == "mlp":
        return MlpActorCritic(obs_dim, action_dim, hidden_size)
    if policy in {"cfc", "lnn"}:
        return CfCActorCritic(obs_dim, action_dim, hidden_size)
    if policy in {"gru", "lstm"}:
        return RecurrentActorCritic(obs_dim, action_dim, hidden_size, kind=policy)
    raise ValueError(f"Unknown policy type: {policy}")
