from typing import List

import io
import torch
import wandb
from matplotlib import pyplot as plt
from PIL import Image
from tensordict import TensorDict, TensorDictBase

from benchmarl.experiment.callback import Callback
from het_control.callbacks.utils import get_het_model
from het_control.snd import compute_behavioral_distance


class TeamDiversityCallback(Callback):
    """Logs intra-team diversity metrics and plots during evaluation."""

    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        if not rollouts:
            return

        logs = {}
        for group in self.experiment.group_map.keys():
            if len(self.experiment.group_map[group]) <= 1:
                continue

            policy = self.experiment.group_policies[group]
            model = get_het_model(policy)
            if model is None:
                continue

            obs_key = (group, "observation")
            if obs_key not in rollouts[0].keys(include_nested=True):
                continue

            episode_divs = []
            for rollout in rollouts:
                obs = rollout.get(obs_key)
                td_in = TensorDict({model.in_key: obs}, batch_size=obs.shape[:-2])
                with torch.no_grad():
                    td_out = model._forward(
                        td_in, compute_estimate=False, update_estimate=False
                    )
                    actions = td_out.get(model.out_key)
                    if model.probabilistic:
                        actions = actions.chunk(2, -1)[0]
                agent_action_list = list(actions.unbind(dim=-2))
                distance = compute_behavioral_distance(agent_action_list, just_mean=True)
                mean_div = distance.mean().item()
                episode_divs.append(mean_div)

            if not episode_divs:
                continue

            mean_div = float(torch.tensor(episode_divs).mean().item())
            logs[f"eval/team_diversity/{group}/intra_mean"] = mean_div
            logs[f"Visuals/team_diversity_{group}_intra"] = self._plot_diversity(
                episode_divs, group
            )

        if logs:
            self.experiment.logger.log(logs, step=self.experiment.n_iters_performed)

    def _plot_diversity(self, values: List[float], group: str) -> wandb.Image:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(values, marker="o")
        ax.set_title(f"Intra-Team Diversity ({group})")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Pairwise Distance")
        ax.grid(True, linestyle="--", alpha=0.6)

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)
        return wandb.Image(Image.open(buf))
