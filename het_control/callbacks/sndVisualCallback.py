from typing import List
import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
import networkx as nx

from het_control.callback import get_het_model
from het_control.snd import compute_behavioral_distance
from benchmarl.experiment.callback import Callback
from tensordict._td import TensorDict  # Required for the Graph Visualizer

class SNDHeatmapVisualizer:
    def __init__(self, key_name="Visuals/SND_Heatmap"):
        self.key_name = key_name

    def generate(self, snd_matrix, step_count, labels=None):
        # snd_matrix is now GUARANTEED to be a clean 2D Numpy array
        n_agents = snd_matrix.shape[0]
        if labels is None:
            agent_labels = [f"Agent {i+1}" for i in range(n_agents)]
        else:
            agent_labels = labels
        
        # Calculate SND value
        iu = np.triu_indices(n_agents, k=1)
        if len(iu[0]) > 0:
            snd_value = float(np.mean(snd_matrix[iu]))
        else:
            snd_value = 0.0

        fig, ax = plt.subplots(figsize=(6, 5))

        im = ax.imshow(
            snd_matrix,
            cmap="viridis",
            interpolation="nearest",
            vmin=0, vmax=3 
        )

        ax.set_title(f"SND: {snd_value:.3f}  –  Step {step_count}")

        ax.set_xticks(np.arange(n_agents))
        ax.set_yticks(np.arange(n_agents))
        ax.set_xticklabels(agent_labels)
        ax.set_yticklabels(agent_labels)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        fig.colorbar(im, ax=ax, label="Distance")

        for i in range(n_agents):
            for j in range(n_agents):
                val = snd_matrix[i, j]
                # Dynamic text color for visibility
                text_color = "white" if val < 1.0 else "black"
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    color=text_color,
                    fontsize=9, fontweight="bold"
                )

        plt.tight_layout()
        img = wandb.Image(fig)
        plt.close(fig)
        return {self.key_name: img}


class SNDBarChartVisualizer:
    def __init__(self, key_name="Visuals/SND_BarChart"):
        self.key_name = key_name

    def generate(self, snd_matrix, step_count, labels=None):
        n_agents = snd_matrix.shape[0]
        if labels is None:
            labels = [f"A{i+1}" for i in range(n_agents)]
        
        # Create pairs i < j
        pairs = [(i, j) for i in range(n_agents) for j in range(i + 1, n_agents)]
        if not pairs:
            return {}

        pair_values = [float(snd_matrix[i, j]) for i, j in pairs]
        pair_labels = [f"{labels[i]}-{labels[j]}" for i, j in pairs]

        snd_value = float(np.mean(pair_values))

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(pair_labels, pair_values, color="teal")

        ax.set_title(f"SND: {snd_value:.3f}  –  Step {step_count}")
        ax.set_ylabel("Distance")
        ax.set_ylim(0, 3)
        ax.tick_params(axis="x", rotation=45)

        ax.bar_label(bars, fmt="%.2f", padding=3)

        plt.tight_layout()
        img = wandb.Image(fig)
        plt.close(fig)
        return {self.key_name: img}


class SNDGraphVisualizer:
    def __init__(self, key_name="Visuals/SND_NetworkGraph"):
        self.key_name = key_name

    def generate(self, snd_matrix, step_count, labels=None):
        n_agents = snd_matrix.shape[0]
        if labels is None:
            labels = [f"A{i+1}" for i in range(n_agents)]

        pairs = [(i, j) for i in range(n_agents) for j in range(i + 1, n_agents)]
        if not pairs:
            return {}

        pair_values = [float(snd_matrix[i, j]) for i, j in pairs]
        snd_value = float(np.mean(pair_values))

        fig = plt.figure(figsize=(7, 7))
        G = nx.Graph()

        for i, j in pairs:
            G.add_edge(i, j, weight=float(snd_matrix[i, j]))

        pos = nx.spring_layout(G, seed=42)
        weights = [G[u][v]['weight'] for u, v in G.edges()]

        nx.draw_networkx_nodes(G, pos, node_size=750, node_color='lightblue')
        
        label_mapping = {i: labels[i] for i in range(n_agents)}
        nx.draw_networkx_labels(G, pos, labels=label_mapping, font_size=12, font_weight='bold')

        edges = nx.draw_networkx_edges(
            G, pos,
            edge_color=weights,
            edge_cmap=plt.cm.viridis,
            width=2,
            edge_vmin=0, edge_vmax=3
        )

        edge_labels = {(i, j): f"{snd_matrix[i, j]:.2f}" for i, j in pairs}
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels,
            font_color='black', font_size=9, font_weight='bold'
        )

        plt.colorbar(edges, label='Distance')
        plt.title(f"SND: {snd_value:.3f}  –  Step {step_count}", fontsize=14)
        plt.axis('off')

        img = wandb.Image(fig)
        plt.close(fig)
        return {self.key_name: img}


class SNDVisualizationManager:
    """
    Manages the individual visualizers and handles ALL data cleaning centrally.
    """
    def __init__(self):
        self.visualizers = [
            SNDHeatmapVisualizer(),
            SNDBarChartVisualizer(),
            SNDGraphVisualizer()
        ]

    def _prepare_matrix(self, snd_matrix):
        """
        Robustly converts and reshapes matrix.
        Fixes crash by Symmetrizing (Broadcasting) BEFORE accessing diagonals.
        """
        # 1. Convert to Numpy
        if hasattr(snd_matrix, "detach"):
            snd_matrix = snd_matrix.detach().cpu().numpy()
        elif not isinstance(snd_matrix, np.ndarray):
            snd_matrix = np.array(snd_matrix)

        # 2. "Peel" dimensions until we hit 2D
        # This turns (1, 2, 2) -> (2, 2) and (1, 2, 1) -> (2, 1)
        while snd_matrix.ndim > 2:
            snd_matrix = snd_matrix[0]

        # 3. Handle 1D edge case (if squeeze happened upstream)
        if snd_matrix.ndim == 1:
            # Try to reshape to square, or expand dims
            size = snd_matrix.shape[0]
            n_agents = int(np.sqrt(size))
            if n_agents * n_agents == size:
                snd_matrix = snd_matrix.reshape(n_agents, n_agents)
            else:
                # Treat as column vector (N, 1)
                snd_matrix = snd_matrix[:, None]

        # 4. Create copy
        snd_matrix = snd_matrix.copy()

        # 5. FIX: Enforce Symmetry FIRST
        # If input is (2, 1), this line broadcasts it: (2, 1) + (1, 2) = (2, 2)
        # This automatically "expands" the missing dimension.
        snd_matrix = (snd_matrix + snd_matrix.T) / 2.0

        # 6. NOW set diagonals (Safe because matrix is guaranteed square now)
        n = snd_matrix.shape[0]
        if n > 0:
            for i in range(n):
                snd_matrix[i, i] = 0.0
        
        return snd_matrix

    def generate_all(self, snd_matrix, step_count, labels=None):
        # Clean the matrix ONCE here
        clean_matrix = self._prepare_matrix(snd_matrix)
        
        all_plots = {}
        for visualizer in self.visualizers:
            try:
                # Pass the clean matrix to all visualizers
                plots = visualizer.generate(clean_matrix, step_count, labels=labels)
                all_plots.update(plots)
            except Exception as e:
                print(f"Error generating {visualizer.__class__.__name__}: {e}")
                # Optional: Print shape to help debug if it fails again
                print(f"Failed Matrix Shape: {clean_matrix.shape}")
        return all_plots
    
class SNDVisualizerCallback(Callback):
    """
    Computes the SND matrix for all agents across all groups and logs visualizations.
    """
    def __init__(self):
        super().__init__()
        self.group_models = []
        self.all_probabilistic = True
        self.viz_manager = SNDVisualizationManager()

    def on_setup(self):
        if not self.experiment.group_policies:
            print("\nWARNING: No group policies found. SND Visualizer disabled.\n")
            return

        self.group_models = []
        self.all_probabilistic = True
        for group, policy in self.experiment.group_policies.items():
            model = get_het_model(policy)
            if model is None:
                continue
            self.group_models.append((group, model))
            self.all_probabilistic = self.all_probabilistic and model.probabilistic

        if not self.group_models:
            print("\nWARNING: Could not extract any HetModel. Visualizer disabled.\n")

    def _get_agent_actions_for_rollout(self, rollout, group, model):
        obs = rollout.get((group, "observation"))
        actions = []
        for i in range(model.n_agents):
            temp_td = TensorDict(
                {(group, "observation"): obs},
                batch_size=obs.shape[:-1],
            )
            action_td = model._forward(temp_td, agent_index=i, compute_estimate=False)
            actions.append(action_td.get(model.out_key))
        return actions

    def _pairwise_to_matrix(self, pairwise, n_agents):
        pairwise = np.asarray(pairwise).reshape(-1)
        matrix = np.zeros((n_agents, n_agents), dtype=float)
        idx = 0
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                value = float(pairwise[idx])
                matrix[i, j] = value
                matrix[j, i] = value
                idx += 1
        return matrix

    def on_evaluation_end(self, rollouts: List[TensorDict]):
        if not self.group_models:
            return

        logs_to_push = {}
        first_rollout_snd_matrix = None
        labels = None

        with torch.no_grad():
            for i, r in enumerate(rollouts):
                if i > 0:
                    break

                agent_actions = []
                labels = []
                for group, model in self.group_models:
                    if (group, "observation") not in r.keys(include_nested=True):
                        continue
                    group_actions = self._get_agent_actions_for_rollout(r, group, model)
                    agent_actions.extend(group_actions)
                    labels.extend([f"{group}_A{i+1}" for i in range(model.n_agents)])

                if len(agent_actions) <= 1:
                    continue

                pairwise_distances = compute_behavioral_distance(
                    agent_actions, just_mean=not self.all_probabilistic
                )
                if pairwise_distances.ndim > 1:
                    pairwise_distances = pairwise_distances.mean(dim=0)

                first_rollout_snd_matrix = self._pairwise_to_matrix(
                    pairwise_distances.cpu().numpy(), len(agent_actions)
                )

        if first_rollout_snd_matrix is not None:
            visual_logs = self.viz_manager.generate_all(
                snd_matrix=first_rollout_snd_matrix,
                step_count=self.experiment.n_iters_performed,
                labels=labels,
            )
            logs_to_push.update(visual_logs)
            self.experiment.logger.log(
                logs_to_push, step=self.experiment.n_iters_performed
            )
