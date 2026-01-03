#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.


import torch
from tensordict import TensorDictBase, TensorDict
from torchrl.envs import EnvBase

from het_control.callbacks.utils import get_het_model
from het_control.snd import compute_behavioral_distance


def render_callback(experiment, env: EnvBase, data: TensorDictBase):
    """
    Render callback used in the Multi-Agent Navigation scenario to visualize the
    diversity distribution under the evaluation rendering.

    """
    if "agents" in experiment.group_policies:
        policy = experiment.group_policies["agents"]
    else:
        policy = next(iter(experiment.group_policies.values()))
    model = get_het_model(policy)
    env_index = 0
    group_name = model.in_key[0] if isinstance(model.in_key, tuple) else "agents"

    if group_name not in env.group_map:
        return env.render(mode="rgb_array", visualize_when_rgb=False, env_index=env_index)

    group_size = len(env.group_map[group_name])
    if group_size <= 1:
        return env.render(mode="rgb_array", visualize_when_rgb=False, env_index=env_index)

    def snd(pos):
        """
        Given a position, this function returns the SND of the policies in that observation
        """
        obs = env.scenario.observation_from_pos(
            torch.tensor(pos, device=model.device), env_index=env_index
        )
        obs = obs.view(-1, group_size, obs.shape[-1]).to(torch.float)
        obs_td = TensorDict(
            {group_name: TensorDict({"observation": obs}, obs.shape[:2])},
            obs.shape[:1],
        )

        agent_actions = []
        for i in range(model.n_agents):
            agent_actions.append(
                model._forward(obs_td, agent_index=i).get(model.out_key)
            )
        
        # This function now correctly returns a tensor of shape [batch_size, n_pairs].
        pairwise_distances = compute_behavioral_distance(
            agent_actions,
            just_mean=True,
        )
        
        # We can now simply take the mean across the pairs.
        avg_distance = pairwise_distances.mean(dim=-1)

        # Reshape to the [batch_size, 1] column vector required by the renderer.
        return avg_distance.view(-1, 1)

    return env.render(
        mode="rgb_array",
        visualize_when_rgb=False,
        plot_position_function=snd,
        plot_position_function_range=1.5,
        plot_position_function_cmap_alpha=0.5,
        env_index=env_index,
        plot_position_function_precision=0.05,
        plot_position_function_cmap_range=[0.0, 1.0],
    )
