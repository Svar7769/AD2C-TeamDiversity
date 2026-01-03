#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, MISSING, field
from typing import Type, Sequence, Optional, Dict, Any, Union

import torch
from tensordict import TensorDictBase
from torch import nn

from benchmarl.models.common import ModelConfig
from het_control.models.het_control_mlp_empirical import HetControlMlpEmpirical
from het_control.utils import overflowing_logits_norm


@dataclass
class TeamDevConfig:
    beta: float = 1.0
    pooling: str = "mean"  # mean | sum | max


@dataclass
class AgentDevConfig:
    zero_mean: bool = True


class HetControlMlpEmpiricalTeam(HetControlMlpEmpirical):
    def __init__(
        self,
        activation_class: Type[nn.Module],
        num_cells: Sequence[int],
        desired_snd: float,
        probabilistic: bool,
        scale_mapping: Optional[str],
        tau: float,
        bootstrap_from_desired_snd: bool,
        process_shared: bool,
        team_dev: Union[TeamDevConfig, Dict[str, Any], None] = None,
        agent_dev: Union[AgentDevConfig, Dict[str, Any], None] = None,
        **kwargs,
    ):
        super().__init__(
            activation_class=activation_class,
            num_cells=num_cells,
            desired_snd=desired_snd,
            probabilistic=probabilistic,
            scale_mapping=scale_mapping,
            tau=tau,
            bootstrap_from_desired_snd=bootstrap_from_desired_snd,
            process_shared=process_shared,
            **kwargs,
        )

        team_dev = team_dev or {}
        agent_dev = agent_dev or {}
        if isinstance(team_dev, TeamDevConfig):
            team_dev = {"beta": team_dev.beta, "pooling": team_dev.pooling}
        if isinstance(agent_dev, AgentDevConfig):
            agent_dev = {"zero_mean": agent_dev.zero_mean}

        self.team_dev_beta = float(team_dev.get("beta", 1.0))
        self.team_dev_pooling = team_dev.get("pooling", "mean")
        self.agent_dev_zero_mean = bool(agent_dev.get("zero_mean", True))

        hidden_dim = self.output_features // 2 if self.probabilistic else self.output_features
        self.team_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        ).to(self.device)

    def _forward(
        self,
        tensordict: TensorDictBase,
        agent_index: int = None,
        update_estimate: bool = True,
        compute_estimate: bool = True,
    ) -> TensorDictBase:
        input = tensordict.get(self.in_key)
        shared_out = self.shared_mlp.forward(input)
        if agent_index is None:
            agent_out = self.agent_mlps.forward(input)
        else:
            agent_out = self.agent_mlps.agent_networks[agent_index].forward(input)

        shared_out = self.process_shared_out(shared_out)

        if (
            self.desired_snd > 0
            and torch.is_grad_enabled()
            and compute_estimate
            and self.n_agents > 1
        ):
            distance = self.estimate_snd(input)
            if update_estimate:
                self.estimated_snd[:] = distance.detach()
        else:
            distance = self.estimated_snd
        if self.desired_snd == 0:
            scaling_ratio = 0.0
        elif (
            self.desired_snd == -1
            or distance.isnan().any()
            or self.n_agents == 1
        ):
            scaling_ratio = 1.0
        else:
            scaling_ratio = torch.where(
                distance != self.desired_snd,
                self.desired_snd / distance,
                1,
            )

        if self.probabilistic:
            shared_loc, shared_scale = shared_out.chunk(2, -1)
            base_feat = shared_loc
        else:
            base_feat = shared_out

        agent_devs = agent_out
        if self.agent_dev_zero_mean:
            agent_devs = agent_devs - agent_devs.mean(dim=-2, keepdim=True)

        if self.team_dev_pooling == "mean":
            g = torch.mean(base_feat, dim=-2)
        elif self.team_dev_pooling == "sum":
            g = torch.sum(base_feat, dim=-2)
        elif self.team_dev_pooling == "max":
            g = torch.max(base_feat, dim=-2).values
        else:
            raise ValueError(f"Unknown team pooling: {self.team_dev_pooling}")

        team_dev = self.team_mlp(g)
        team_dev_broadcast = team_dev.unsqueeze(-2).expand_as(base_feat)

        if self.probabilistic:
            agent_loc = base_feat + agent_devs * scaling_ratio + team_dev_broadcast * self.team_dev_beta
            out_loc_norm = overflowing_logits_norm(
                agent_loc, self.action_spec[self.agent_group, "action"]
            )
            out = torch.cat([agent_loc, shared_scale], dim=-1)
        else:
            out = base_feat + agent_devs * scaling_ratio + team_dev_broadcast * self.team_dev_beta
            out_loc_norm = overflowing_logits_norm(
                out, self.action_spec[self.agent_group, "action"]
            )

        tensordict.set(
            (self.agent_group, "estimated_snd"),
            self.estimated_snd.expand(tensordict.get_item_shape(self.agent_group)),
        )
        tensordict.set(
            (self.agent_group, "scaling_ratio"),
            (
                torch.tensor(scaling_ratio, device=self.device).expand_as(out)
                if not isinstance(scaling_ratio, torch.Tensor)
                else scaling_ratio.expand_as(out)
            ),
        )
        tensordict.set((self.agent_group, "logits"), out)
        tensordict.set((self.agent_group, "out_loc_norm"), out_loc_norm)

        tensordict.set(self.out_key, out)

        return tensordict


@dataclass
class HetControlMlpEmpiricalTeamConfig(ModelConfig):
    activation_class: Type[nn.Module] = MISSING
    num_cells: Sequence[int] = MISSING

    desired_snd: float = MISSING
    tau: float = MISSING
    bootstrap_from_desired_snd: bool = MISSING
    process_shared: bool = MISSING

    probabilistic: Optional[bool] = MISSING
    scale_mapping: Optional[str] = MISSING

    team_dev: TeamDevConfig = field(default_factory=TeamDevConfig)
    agent_dev: AgentDevConfig = field(default_factory=AgentDevConfig)

    @staticmethod
    def associated_class():
        return HetControlMlpEmpiricalTeam
