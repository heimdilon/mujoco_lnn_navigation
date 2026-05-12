from __future__ import annotations

import torch
from torch import nn

from ncps.torch import CfC as NcpsCfC
from ncps.wirings import AutoNCP

from mujoco_lnn_nav.models.policies import BaseActorCritic, _squashed_normal_action


class NcpCfCActorCritic(BaseActorCritic):
    """NCP-wired CfC policy: sparse sensory→inter→command→motor topology.

    Mimics ``CfCActorCritic`` API (forward / forward_recurrent / forward_sequence /
    initial_state / act_recurrent / reset_state_indices) so the existing BC + DAgger
    pipeline runs unchanged. The recurrent backbone is a single ``ncps.torch.CfC``
    instance whose ``units`` argument is an ``AutoNCP`` wiring object — this swaps the
    dense backbone for a C. elegans-inspired sparse circuit while keeping the
    continuous-time dynamics of the underlying CfC cell.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        encoder_dim: int = 64,
        ncp_units: int = 48,
        ncp_output_size: int = 16,
        sparsity_level: float = 0.5,
        wiring_seed: int = 22222,
        dt: float = 0.08,
        cfc_mode: str = "default",
    ):
        super().__init__()
        self.kind = "ncp_cfc"
        self.policy_impl = "ncps.torch.CfC + AutoNCP"
        self.dt = dt
        self.encoder_dim = encoder_dim
        self.ncp_units = ncp_units
        self.ncp_output_size = ncp_output_size
        self.sparsity_level = sparsity_level
        self.wiring_seed = wiring_seed

        self.input = nn.Sequential(nn.Linear(obs_dim, encoder_dim), nn.Tanh())

        self.wiring = AutoNCP(
            units=ncp_units,
            output_size=ncp_output_size,
            sparsity_level=sparsity_level,
            seed=wiring_seed,
        )

        # Wiring drives topology — no backbone_units / backbone_layers here.
        self.rnn = NcpsCfC(
            encoder_dim,
            self.wiring,
            return_sequences=True,
            batch_first=True,
            mode=cfc_mode,
        )

        self.actor = nn.Linear(ncp_output_size, action_dim)
        self.critic = nn.Linear(ncp_output_size, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def _timespans(self, seq: int, device: torch.device) -> torch.Tensor:
        return torch.full((1, seq), self.dt, dtype=torch.float32, device=device)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.input(obs).unsqueeze(1)
        ts = self._timespans(1, enc.device)
        output, _ = self.rnn(enc, timespans=ts)
        features = output[:, -1]
        return self.actor(features), self.critic(features)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((batch_size, self.ncp_units), dtype=torch.float32, device=device)

    def reset_state_indices(self, state: torch.Tensor, indices: list[int]) -> torch.Tensor:
        if indices:
            state[indices, :] = 0.0
        return state

    def forward_recurrent(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enc = self.input(obs).unsqueeze(1)
        ts = self._timespans(1, enc.device)
        output, next_state = self.rnn(enc, state, timespans=ts)
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
        encoded = self.input(sequence)
        ts = self._timespans(encoded.shape[1], encoded.device)
        output, _ = self.rnn(encoded, timespans=ts)
        mean = self.actor(output)
        value = self.critic(output).squeeze(-1)
        if is_unbatched:
            mean = mean.squeeze(0)
            value = value.squeeze(0)
        return mean, value

    def save_wiring_diagram(self, path: str) -> None:
        import matplotlib.pyplot as plt

        self.wiring.draw_graph(layout="shell")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
