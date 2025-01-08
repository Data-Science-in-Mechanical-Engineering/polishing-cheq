import os
import numpy as np
import wandb
import datetime
from typing import Union, Type
import logging
import warnings

from utils.timer_utils import Timer
from environments.mujoco_gym_env import MujocoGymEnv

import gymnasium as gym
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from omegaconf import DictConfig, OmegaConf
import tempfile


def _recursive_flatten_dict(d: dict):
    keys, values = [], []
    for key, value in d.items():
        if isinstance(value, dict):
            sub_keys, sub_values = _recursive_flatten_dict(value)
            keys += [f"{key}/{k}" for k in sub_keys]
            values += sub_values
        else:
            keys.append(key)
            values.append(value)
    return keys, values

def create_rollout_stats(
    env: Type[gym.Env],
    lambda_lst: list, 
    uncertainty_lst: list, 
    hybrid_a_lst: list,
    rl_a_lst: list,
    info_dict: dict,
    learning_started: bool, 
    timer: Timer, 
    step: int
) -> dict:
    """
    IMPROVEMENT PLANNED: put all metrics in one dict for better argument structure, put env in to range histograms of actions

    Based on the input lists, creates a dictionary that can be taken as the input
    for the request.
    Further, distinguishes between including stats for uncertainty or not (if no information is included).

    Args:
        lambda_lst (list): The episodes list of lambdas.
        uncertainty_lst (list): The episodes list of uncertainties.
        hybrid_a_lst (list): The episodes list of hybrid actions.
        rl_a_lst (list): The episodes list of rl actions.
        info_dict (dict): The dictionary of the last step of the environment.
        learning_started (bool): For tracking purposes.
        timer (Timer): The timer for each task.
        step (int): The current actor_step.

    Returns:
        dict: Transformed dictionary for tracking actor behavior in wandb.
    """
    stats = {
        "rollout/lambda_dist": wandb.Histogram(np_histogram=np.histogram(lambda_lst, bins=20)),
        "rollout/learning_started": int(learning_started), 
        "rollout/episodic_length": info_dict.get("episodic_length", None),
        "rollout/fails": info_dict.get("fails_count", None),
        "rollout/return": info_dict.get("return", None),
        "rollout/min_y_pos": info_dict.get("min_y_pos", None),
        "timer": timer.get_average_times(), 
        "steps": {"actor_step": step},
    }

    if env.unwrapped.action_space.shape == (1,):
        # actually have to verify if this holds for the CartPoleEnv
        # further I should include logic for higher shapes so plot as well
        stats_action = {
            "rollout/hybrid_a_dist": wandb.Histogram(np_histogram=np.histogram(hybrid_a_lst, bins=20, range=(-10,10))),
            "rollout/rl_a_dist": wandb.Histogram(np_histogram=np.histogram(rl_a_lst, bins=20, range=(-10,10))),
        }
    else:
        # expect to visualize the positional action with first 3 action dimensions
        stats_action = {
            "rollout/hybrid_a_dist": wandb.Histogram(np_histogram=np.histogram(np.linalg.norm(np.array(hybrid_a_lst)[:,:3], axis=1), bins=20)),
            "rollout/rl_a_dist": wandb.Histogram(np_histogram=np.histogram(np.linalg.norm(np.array(rl_a_lst)[:,:3], axis=1), bins=20)),
        }
    stats = {**stats, **stats_action}

    if set(uncertainty_lst) != {None}:
        # if we are using a measure of uncertainty
        stats_uncertainty = {
            "rollout/uncertainty_dist": wandb.Histogram(np_histogram=np.histogram(uncertainty_lst, bins=20)),
            "rollout/uncertainty_avg": np.mean(np.array(uncertainty_lst))
        }
        stats = {**stats, **stats_uncertainty}
    else:
        pass

    if isinstance(env.unwrapped, MujocoGymEnv):
        # for all robot simulation environments
        stats_rewards = {f"rollout/{key}": val for (key, val) in info_dict.get("reward_info").items()}
        stats_robot = {
            "rollout/num_wiped_markers": info_dict.get("num_wiped_markers", None),
            "rollout/force_dist": wandb.Histogram(np_histogram=np.histogram(info_dict.get("episode_forces", None), bins=20)),
        }
        mean_action = np.mean(np.array(hybrid_a_lst), axis=0)
        rescaled_mean_action = env.unwrapped.rescaling_func(mean_action)
        stats_gains = {
            "rollout/stiff_x": rescaled_mean_action[-7],
            "rollout/stiff_y": rescaled_mean_action[-6],
            "rollout/stiff_z": rescaled_mean_action[-5],
            "rollout/stiff_rot_x": rescaled_mean_action[-4],
            "rollout/stiff_rot_y": rescaled_mean_action[-3],
            "rollout/stiff_rot_z": rescaled_mean_action[-2],
            "rollout/damp_ratio": rescaled_mean_action[-1],
        }
        
        stats = {**stats, **stats_robot, **stats_rewards, **stats_gains}

    return stats


class WandBLogger(object):
    
    def __init__(self, wandb_config, wandb_output_dir=None, debug=False):
        wandb.require("service")
        self.config = wandb_config
        # extract the runs from the specified wandb project and group
        if self.config.entity is not None:
            runs_path = f"{self.config.entity}/{self.config.project}"
        else:
            runs_path = f"{self.config.project}"

        api = wandb.Api()
        runs = api.runs(path=runs_path, filters={"group": self.config.group})

        # name run according to the job type and number of jobs in the group
        run_numbers = [int(run.name.split("_")[-1]) for run in runs if run.name.startswith(self.config.job_type)]
        run_name = f"{self.config.job_type}_{max(run_numbers, default=0) + 1}"
        run_id = f"{run_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if wandb_output_dir is None:
            wandb_output_dir = tempfile.mkdtemp()

        mode = "disabled" if wandb_config.debug else "online"

        # initialize the wandb run
        self.run = wandb.init(project=self.config.project, 
                              entity=self.config.entity, 
                              group=self.config.group, 
                              name=run_name, 
                              id=run_id, 
                              job_type=self.config.job_type,
                              dir=wandb_output_dir,
                              mode=mode)
        
    def prepare_logger(self, cfg: Union[dict, DictConfig]):
        """
        Transform config to dict-like structure and log to wandb.
        Further, adds tags to the run based on the config file if create_tags is active.
        """
        # log config file
        if isinstance(cfg, dict):
            config = cfg
        elif isinstance(cfg, DictConfig):
            config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        else:
            raise ValueError("Config must be either a dict or a DictConfig object.")

        self.run.config.update(config)

        # create tags from config file:
        tags = self._create_tags(config)
        self._add_tags(tags)

        # defines all metrics depending on configuration
        if config["rl"]["_target_"].split(".")[-1] == "SAC":
            using_esemble = False
        elif config["rl"]["_target_"].split(".")[-1] == "SACEnsemble":
            using_esemble = True
        else:
            raise NotImplementedError("Please specify your RLAgent in the logging_utils!")

        if config["environment"]["name"] == "sim_robot":
            robot_env = True
        else:
            robot_env = False

        self._define_all_metrics(using_esemble, robot_env)


    def log(self, data: dict, commit: bool = True, step: int = None):
        """
        Logs data to wandb.
        Also transforms nested dictionaries into flatten ones.
        """
        data_flat = _recursive_flatten_dict(data)
        data = {k: v for k, v in zip(*data_flat)}
        self.run.log(data, commit=commit, step=step)


    def log_video(self, data: np.ndarray, commit: bool = True, step: int = None):
        """
        Logs a video of the environment to wandb, based on the provided frames in data.

        Args:
            data (np.ndarray): The data of the frames
            commit (bool, optional): Whether to directly commit or wait. Defaults to True.
            step (int, optional): A possible step. Defaults to None.
        """
        # get rid of warnings and matplotlogger.info
        warnings.filterwarnings("ignore", category=UserWarning)
        matplotlogger = logging.getLogger("matplotlib.animation")
        matplotlogger.setLevel(logging.ERROR)

        # create folder
        filepath = os.path.abspath(os.path.join(self.run.dir,"videos"))
        if not os.path.isdir(filepath):
            os.makedirs(filepath)

        # record video
        video_file = os.path.join(filepath,f"{step}.mp4")


        fig, ax = plt.subplots(figsize=(6, 4))
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        ax.axis('off')

        img = plt.imshow(data[0], aspect='auto')

        def animate(frame):
            img.set_data(data[frame])
            return [img]

        anim = FuncAnimation(fig, animate, frames=min([len(data),900]), interval=20)
        plt.close()
        anim.save(video_file, fps=30)

        self.run.log({"video": wandb.Video(video_file, format="gif")}, step=step, commit=commit)


    def _add_tags(self, lst: list) -> None:
        """
        Adds tags to the current wandb run.

        Args:
            lst (list): The list to be added to the tags
        """
        for item in lst:
            self.run.tags = self.run.tags + (item,)


    def _create_tags(self, config: dict) -> list:
        """
        Creates the tags based on the config file.

        Args:
            config (dict): The config file of the current run

        Returns:
            list: Outputs the list of strings representing the tags.
        """
        tags = [
            "EnvName: " + config["environment"]["name"],
            "HybridAgent: " + config["hybrid"]["_target_"].split(".")[-1],
            "RLCont: " + config["rl"]["_target_"].split(".")[-1],
        ]

        if config["nominal"]["_target_"].split(".")[-1] == "NominalCartPole":
            if config["nominal"]["silly"] == True:
                tags.append("NomCont: SillyCartPole")
                tags.append("SillyAction: " + str(config["nominal"]["action_skew"]))
            else:
                tags.append("NomCont: LQRCartPole")
                if config["nominal"]["action_skew"]:
                    tags.append("ActionSkew: " + str(config["nominal"]["action_skew"]))
        else:
            tags.append("NomCont: " + config["nominal"]["_target_"].split(".")[-1])
        
        if config["hybrid"]["_target_"].split(".")[-1] == "HybridAgent":
            # for fixed hybrid weighting
            tags.append("MixParam: " + str(config["hybrid"]["eta"]))
        elif config["hybrid"]["_target_"].split(".")[-1] == "CHEQAgent":
            # for CHEQAgent (adaptive weighting)
            tags.append("UTD: " + str(config["training"]["async_learner"]["batch_size"][0]))
            tags.append("EnsembleSize: " + str(config["rl"]["ensemble_size"]))
            tags.append("BernoulliKappa: " + str(config["training"]["async_learner"]["kappa"]))

        return tags


    def _define_all_metrics(self, using_esemble: bool, robot_env: bool):
        """
        Defines the metrics which is helpful for automatic plotting with distinct axes.
        """
        # steps
        self.run.define_metric("steps/actor_step")
        self.run.define_metric("steps/learner_step")

        # buffer
        self.run.define_metric("buffer/position_percentage", step_metric="steps/learner_step")

        # agent_train
        self.run.define_metric("agent_train/actor_entropy", step_metric="steps/learner_step")
        self.run.define_metric("agent_train/actor_loss", step_metric="steps/learner_step")
        self.run.define_metric("agent_train/entropy_coefficient", step_metric="steps/learner_step")
        self.run.define_metric("agent_train/q_loss", step_metric="steps/learner_step")
        self.run.define_metric("agent_train/entropy_coefficient_loss", step_metric="steps/learner_step")
        self.run.define_metric("agent_train/target_entropy", step_metric="steps/learner_step")
        
        # evaluation
        self.run.define_metric("eval/learning_started", step_metric="steps/actor_step")
        self.run.define_metric("eval/return", step_metric="steps/actor_step")
        self.run.define_metric("eval/lambda_dist", step_metric="steps/actor_step")
        self.run.define_metric("eval/episodic_length", step_metric="steps/actor_step")

        # rollout
        self.run.define_metric("rollout/learning_started", step_metric="steps/actor_step")
        self.run.define_metric("rollout/return", step_metric="steps/actor_step")
        self.run.define_metric("rollout/fails", step_metric="steps/actor_step")
        self.run.define_metric("rollout/episodic_length", step_metric="steps/actor_step")
        self.run.define_metric("rollout/rl_a_dist", step_metric="steps/actor_step")
        self.run.define_metric("rollout/hybrid_a_dist", step_metric="steps/actor_step")
        self.run.define_metric("rollout/lambda_dist", step_metric="steps/actor_step")

        if using_esemble:
            self.run.define_metric("rollout/uncertainty_dist", step_metric="steps/actor_step")
            self.run.define_metric("rollout/uncertainty_avg", step_metric="steps/actor_step")
            self.run.define_metric("agent_train/next_q_values", step_metric="steps/learner_step")
        else:
            self.run.define_metric("agent_train/qf1_values", step_metric="steps/learner_step")
            self.run.define_metric("agent_train/qf1_loss", step_metric="steps/learner_step")
            self.run.define_metric("agent_train/qf2_values", step_metric="steps/learner_step")
            self.run.define_metric("agent_train/qf2_loss", step_metric="steps/learner_step")

        if robot_env:
            self.run.define_metric("rollout/num_wiped_markers", step_metric="steps/actor_step")
            self.run.define_metric("rollout/force_dist", step_metric="steps/actor_step")
            self.run.define_metric("rollout/penalty_force", step_metric="steps/actor_step")
            self.run.define_metric("rollout/penalty_xvel", step_metric="steps/actor_step")
            self.run.define_metric("rollout/penalty_xdist", step_metric="steps/actor_step")
            self.run.define_metric("rollout/penalty_wvel", step_metric="steps/actor_step")
            self.run.define_metric("rollout/penalty_dist_wipe", step_metric="steps/actor_step")
            self.run.define_metric("rollout/reward_contact", step_metric="steps/actor_step")
            self.run.define_metric("rollout/reward_done", step_metric="steps/actor_step")
            self.run.define_metric("rollout/reward_cross_error", step_metric="steps/actor_step")
            self.run.define_metric("rollout/reward_abs_vel_error", step_metric="steps/actor_step")
            self.run.define_metric("rollout/reward_force_error", step_metric="steps/actor_step")
            self.run.define_metric("rollout/reward_dir_error", step_metric="steps/actor_step")
            self.run.define_metric("rollout/stiff_x", step_metric="steps/actor_step")
            self.run.define_metric("rollout/stiff_y", step_metric="steps/actor_step")
            self.run.define_metric("rollout/stiff_z", step_metric="steps/actor_step")
            self.run.define_metric("rollout/stiff_rot_x", step_metric="steps/actor_step")
            self.run.define_metric("rollout/stiff_rot_y", step_metric="steps/actor_step")
            self.run.define_metric("rollout/stiff_rot_z", step_metric="steps/actor_step")
            self.run.define_metric("rollout/damp_ratio", step_metric="steps/actor_step")
            self.run.define_metric("rollout/min_y_pos", step_metric="steps/actor_step")
            self.run.define_metric("eval/num_wiped_markers", step_metric="steps/actor_step")
        else:
            pass
