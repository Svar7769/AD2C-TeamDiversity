# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from pathlib import Path

import sys
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from hydra.core.global_hydra import GlobalHydra

from het_control.callbacks.sndVisualCallback import SNDVisualizerCallback
import benchmarl.models
from benchmarl.algorithms import *
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment
from benchmarl.hydra_config import (
    load_algorithm_config_from_hydra,
    load_experiment_config_from_hydra,
    load_task_config_from_hydra,
    load_model_config_from_hydra,
)
from het_control.callback import *
from het_control.environments.vmas import render_callback
from het_control.models.het_control_mlp_empirical import HetControlMlpEmpiricalConfig
from het_control.models.het_control_mlp_empirical_team import (
    HetControlMlpEmpiricalTeamConfig,
)
from het_control.callbacks.esc_callback import ExtremumSeekingController
from het_control.callbacks.sndESLogger import TrajectorySNDLoggerCallback
from het_control.callbacks.team_diversity_callback import TeamDiversityCallback
# IMPORTANT: Import and run the setup function from your training script
# This registers your custom model "hetcontrolmlpempirical" so it can be loaded.
from het_control.run import setup
setup("vmas/navigation")


if __name__ == "__main__":
    # 1. Define the path to your checkpoint
    # Using Path helps make the path work across different operating systems.
    checkpoint_path = "/home/svarp/Desktop/Projects/ad2c/ControllingBehavioralDiversity/model_checkpoints/navigation_team_ippo/ippo_navigation_team_hetcontrolmlpempiricalteam__fb96521b_26_01_03-00_57_41/checkpoints/checkpoint_1200000.pt"

    # 2. Define a list of settings to override for the evaluation
    # The original task, algorithm, and model are loaded automatically.
    # We only need to specify what we want to change.
    evaluation_overrides = [
        "experiment.render=True",         # Turn on visual rendering
        "experiment.max_n_frames=5000",   # Limit the evaluation to 5000 steps
        "callbacks=[]"                    # Ensure no training callbacks are used
    ]

    # 3. Reload the experiment with the overrides
    print(f"Loading experiment from {checkpoint_path}")
    experiment = Experiment(
        checkpoint_path=checkpoint_path,
        overrides=evaluation_overrides
    )

    # 4. Run the evaluation
    print("Starting evaluation...")
    experiment.evaluate()
    print("Evaluation finished.")