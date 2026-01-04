#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

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
from het_control.models.het_control_mlp_empirical_team import (
    HetControlMlpEmpiricalTeamConfig,
)
from het_control.callbacks.esc_callback import ExtremumSeekingController
from het_control.callbacks.sndESLogger import TrajectorySNDLoggerCallback
from het_control.callbacks.team_diversity_callback import TeamDiversityCallback



def setup(task_name):
    benchmarl.models.model_config_registry.update(
        {
            "hetcontrolmlpempiricalteam": HetControlMlpEmpiricalTeamConfig,
        }
    )
    if task_name == "vmas/navigation_team":
        VmasTask.render_callback = render_callback


def get_experiment(cfg: DictConfig) -> Experiment:
    hydra_choices = HydraConfig.get().runtime.choices
    task_name = hydra_choices.task
    algorithm_name = hydra_choices.algorithm

    setup(task_name)

    print(f"\nAlgorithm: {algorithm_name}, Task: {task_name}")
    print("\nLoaded config:\n")
    print(OmegaConf.to_yaml(cfg))

    algorithm_config = load_algorithm_config_from_hydra(cfg.algorithm)
    experiment_config = load_experiment_config_from_hydra(cfg.experiment)
    task_config = load_task_config_from_hydra(cfg.task, task_name)
    critic_model_config = load_model_config_from_hydra(cfg.critic_model)
    model_config = load_model_config_from_hydra(cfg.model)

    if isinstance(algorithm_config, (MappoConfig, IppoConfig, MasacConfig, IsacConfig)):
        model_config.probabilistic = True
        model_config.scale_mapping = algorithm_config.scale_mapping
        algorithm_config.scale_mapping = "relu"
    else:
        model_config.probabilistic = False

    experiment = Experiment(
        task=task_config,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=cfg.seed,
        config=experiment_config,
        callbacks=[
            # SndCallback(),
            # ExtremumSeekingController(  # assumes single "agents" group
            #             control_group="agents",
            #             initial_snd=0.0,
            #             dither_magnitude=0.2,
            #             dither_frequency_rad_s=1.0,
            #             integral_gain=-0.1,
            #             high_pass_cutoff_rad_s=1.0,
            #             low_pass_cutoff_rad_s=1.0,
            #             sampling_period=1.0
            # ),
            # TrajectorySNDLoggerCallback(control_group="agents"),
            SNDVisualizerCallback(),
            TeamDiversityCallback(),
            NormLoggerCallback(),
            ActionSpaceLoss(
                use_action_loss=cfg.use_action_loss, action_loss_lr=cfg.action_loss_lr
            ),
        ]
        + (
            [
                TagCurriculum(
                    cfg.simple_tag_freeze_policy_after_frames,
                    cfg.simple_tag_freeze_policy,
                )
            ]
            if task_name == "vmas/simple_tag"
            else []
        ),
    )
    return experiment


ABS_CONFIG_PATH = "/home/svarp/Desktop/Projects/ad2c/ControllingBehavioralDiversity/het_control/conf"
CONFIG_NAME = "navigation_team_ippo"
SAVE_PATH = "/home/svarp/Desktop/Projects/ad2c/ControllingBehavioralDiversity/model_checkpoints/navigation_team_ippo/"

save_interval = 600000
desired_snd = -1.0
max_frame = 600000

GlobalHydra.instance().clear()

sys.argv = [
    "dummy.py",
    f"model.desired_snd={desired_snd}",
    f"experiment.max_n_frames={max_frame}",
    f"experiment.checkpoint_interval={save_interval}",
    f"experiment.save_folder={SAVE_PATH}",
]


@hydra.main(version_base=None, config_path=ABS_CONFIG_PATH, config_name=CONFIG_NAME)
def hydra_experiment(cfg: DictConfig) -> None:
    experiment = get_experiment(cfg=cfg)
    experiment.run()


if __name__ == "__main__":
    try:
        hydra_experiment()
    except SystemExit:
        print("Experiment finished successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
