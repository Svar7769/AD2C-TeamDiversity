from itertools import combinations
from typing import List

import torch
from tensordict import TensorDict, TensorDictBase

from benchmarl.experiment.callback import Callback
from het_control.callbacks.utils import get_het_model
from het_control.snd import compute_behavioral_distance, compute_statistical_distance


class TeamDiversityCallback(Callback):
    """Logs intra-team diversity metrics during evaluation."""

    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        if not rollouts:
            return

        logs = {}
        group_actions = {}
        group_action_dims = {}
        group_batch_shapes = {}

        for group in self.experiment.group_map.keys():
            if len(self.experiment.group_map[group]) <= 1:
                continue

            policy = self.experiment.group_policies[group]
            model = get_het_model(policy)
            if model is None:
                continue
            
            # Concatenate observations across all rollouts
            obs = torch.cat(
                [rollout.get((group, "observation")) for rollout in rollouts], dim=0
            )
            
            # Wrap in TensorDict with correct batch size
            td_obs = TensorDict({model.in_key: obs}, batch_size=obs.shape[:-1])
                
            agent_actions = []
            for i in range(model.n_agents):
                with torch.no_grad():
                    # Pass TensorDict to the model to avoid AttributeError
                    td_out = model._forward(td_obs, agent_index=i, compute_estimate=False)
                    actions = td_out.get(model.out_key)
                    agent_actions.append(actions)

            group_actions[group] = agent_actions
            group_action_dims[group] = agent_actions[0].shape[-1]
            group_batch_shapes[group] = agent_actions[0].shape[:-1]
            print(
                f"[TeamDiversityCallback] group={group} action_dim={group_action_dims[group]}"
            )

            # Intra-team diversity (average pairwise distance within group)
            distance = compute_behavioral_distance(agent_actions, just_mean=True)
            logs[f"eval/{group}/intra_team_diversity"] = distance.mean().item()

        # Inter-team diversity (average pairwise distance across teams)
        group_names = list(group_actions.keys())
        for t1, t2 in combinations(group_names, 2):
            if group_action_dims[t1] != group_action_dims[t2]:
                continue
            if group_batch_shapes[t1] != group_batch_shapes[t2]:
                continue

            pair_results = []
            for actions_i in group_actions[t1]:
                for actions_j in group_actions[t2]:
                    pair_results.append(
                        compute_statistical_distance(actions_i, actions_j, just_mean=True)
                    )

            if not pair_results:
                continue

            inter_distance = torch.stack(pair_results, dim=-1)
            logs[f"eval/inter_team_diversity/{t1}_vs_{t2}"] = (
                inter_distance.mean().item()
            )

        # Global system SND (across all agents) when shapes align
        if group_names:
            first_group = group_names[0]
            action_dim = group_action_dims[first_group]
            batch_shape = group_batch_shapes[first_group]

            all_actions = []
            all_match = True
            for group in group_names:
                if group_action_dims[group] != action_dim:
                    all_match = False
                    break
                if group_batch_shapes[group] != batch_shape:
                    all_match = False
                    break
                all_actions.extend(group_actions[group])

            if all_match and len(all_actions) > 1:
                global_distance = compute_behavioral_distance(
                    all_actions, just_mean=True
                )
                logs["eval/system/snd"] = global_distance.mean().item()

        if logs:
            self.experiment.logger.log(logs, step=self.experiment.n_iters_performed)
