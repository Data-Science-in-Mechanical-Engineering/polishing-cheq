# This code was inspired by SERL (https://serl-robot.github.io/)
# https://github.com/rail-berkeley/serl

from agentlace.trainer import TrainerServer
from agents.rl.rl_base import RLBase
import tqdm
from utils.timer_utils import Timer
from utils.data_utils import ReplayBufferDataStore
from utils.logging_utils import WandBLogger
from utils.launcher_utils import make_trainer_config
from environments.mujoco_gym_env import MujocoGymEnv
from stable_baselines3.common.save_util import save_to_pkl, load_from_pkl
from typing import Literal
from pathlib import Path
from omegaconf import OmegaConf
from matplotlib import pyplot as plt
import omegaconf
import wandb
import warnings
import gymnasium as gym
import time
import pandas as pd
import numpy as np
import wandb
import os
import shutil
import torch
import threading


class AsyncTrainer:
    """
    Learner class for training the agent asynchronously to interacting with the environment.
    
    This class implements the learner side of the asynchronous RL setup, where training and interaction with the environment are decoupled 
    to enhance robustness and timing determinism in real-world learning tasks. It implements the base functionality for the learner loop of 
    the RL agent, where it samples transitions from the replay buffer and trains the agent on them while iteratively sending the new weights
    to the actor. Aditionally, it implements functionality for logging training statistics and checkpointing of models.

    Attributes:
        agent: The hybrid agent to be trained.
        wandb_logger: An instance of the WandBLogger class for logging training statistics.
        learner_config: Configuration for the learner.
        replay_buffer: The replay buffer to store and sample transitions from.
        eval_episode_count: Counter for the number of evaluation episodes run so far.
        update_steps: Counter for the number of training steps taken so far.
        training_paused: Flag to pause training when requested by the actor.
        trainer_server: The server to receive requests from the actor.
    """

    def __init__(
            self, 
            agent: RLBase, 
            env: gym.Env, 
            wandb_logger: WandBLogger, 
            learner_config
    ) -> None:
        """
        Initializes the asynchronous learner.
        
        Args:
            agent: The RL agent to be trained.
            env: The environment to interact with.
            wandb_logger: An instance of the WandBLogger class for logging training statistics.
            learner_config: Configuration for the learner.
        """
        
        self.agent = agent
        self.env = env
        self.wandb_logger = wandb_logger
        self.learner_config = learner_config

        self.replay_buffer = ReplayBufferDataStore(capacity=self.learner_config.buffer_size, 
                                                   observation_space=self.env.observation_space, 
                                                   action_space=self.env.action_space,
                                                   device=self.learner_config.device,
                                                   ensemble_size=self.learner_config.ensemble_size,
                                                   kappa=self.learner_config.kappa)

        self.eval_episode_count = 0
        self.update_steps = 0
        self.updates_left = 0
        self.episodes_received = 0
        self.episodes_trained = 0
        self.training_paused = False
        self.finished_run = False

        # define batch_size in readable manner
        if isinstance(self.learner_config.batch_size, omegaconf.listconfig.ListConfig):
            self.batch_size = OmegaConf.to_container(self.learner_config.batch_size, resolve=True)
        else:
            self.batch_size = self.learner_config.batch_size

        self.trainer_server = TrainerServer(config=make_trainer_config(),
                                            request_callback=self.actor_callbacks)
        
        # start server for the actor
        self.trainer_server.register_data_store("actor_env", self.replay_buffer)
        self.trainer_server.start(threaded=True)

        warnings.filterwarnings(action="ignore", category=FutureWarning)


    def actor_callbacks(
            self, 
            type: Literal["send-stats", "send-eval-metrics", "pause-training", "continue-training", "store-best-model"], 
            payload: dict
    ) -> None:
        """
        Callback function for handling requests from the actor.
        
        This method implements the required callback functions for handling requests from the actor. These include:
        - Logging dict-like stats from the actor to WandB.
        - Logging evaluation metrics from the actor's evaluation runs to WandB.
        - Pausing training when requested by the actor.
        - Continuing training when requested by the actor.
        - Storing the best model when requested by the actor.
        
        Args:
            type: The type of request to handle.
            payload: The payload of the request to handle.
        """

        if type == "send-stats":
            # log payload to wandb
            self.wandb_logger.log(payload)
            
            # take actor rollout length and save in instance for updates per data input
            last_episodic_length = payload.get("rollout/episodic_length", None)
            self.updates_left = last_episodic_length
            self.episodes_received += 1
            # wait some time for training to continue
            time.sleep(1e-3)
            self.training_paused = False

        elif type == "send-eval-metrics":
            # increment episode count
            self.eval_episode_count += 1

            # get video_frames from the payload as list
            video_frames = payload.pop("video_frames", None)
            # expects that the payload is structured as follows: dict(eval_episodes: dict(episode_step: info_dict(metrics: values)))
            df_lst = []
            for episodes in payload.keys():
                # transform payload to dataframe with rows being the steps and columns being the metrics
                df_lst.append(pd.DataFrame(payload.get(episodes, None)).transpose())
            
            learning_started = np.amax([df["learning_started"].max() for df in df_lst])
            episodic_length = np.mean([df["episodic_length"].max() for df in df_lst])
            episodic_return = np.mean([df["return"].iloc[-1] for df in df_lst])
            # for histogram take all lambda values of the dict
            lambda_dist = [df.loc[step, "mixing_parameter"] for df in df_lst for step in df.index]

            eval_data = {
                "eval/learning_started": learning_started,
                "eval/episodic_length": episodic_length,
                "eval/return": episodic_return,
                "eval/lambda_dist": wandb.Histogram(lambda_dist), 
            }

            # for robotic environments
            if isinstance(self.env.unwrapped, MujocoGymEnv):
                
                warnings.filterwarnings(action="ignore", category=UserWarning)

                fig_force, axs_force = plt.subplots()
                try:
                    axs_force.plot(np.array([[df.loc[step, "force"] for step in df.index] for df in df_lst]).T.mean(axis=1))
                    axs_force.axhline(y=self.env.unwrapped.target_force, xmin=0, xmax=episodic_length, color="red", linestyle="--")
                    axs_force.set_ybound(lower=self.env.unwrapped.target_force-2.0, upper=self.env.unwrapped.target_force+2.0)
                except ValueError:
                    print(f"Force error due to ValueError: {[[df.loc[step, 'force'] for step in df.index] for df in df_lst]}")
                    raise
                axs_force.set_ylabel("force")
                axs_force.set_xlabel("episode_step")

                fig_vel, axs_vel = plt.subplots()
                axs_vel.plot(np.array([[np.linalg.norm(np.array([df.loc[step, "x_vel"], df.loc[step, "y_vel"], df.loc[step, "z_vel"]])) for step in df.index] for df in df_lst]).T.mean(axis=1))
                axs_vel.axhline(y=self.env.unwrapped.target_wvel, xmin=0, xmax=episodic_length, color="red", linestyle="--")
                axs_vel.set_ybound(lower=self.env.unwrapped.target_wvel-0.015, upper=self.env.unwrapped.target_wvel+0.015)
                #axs_vel.plot(np.array([[((df.loc[step+1, "state"] - df.loc[step, "state"])**2).sum()**0.5 for step in df.index if step+1 in df.index] for df in df_lst]).T.mean(axis=1))
                axs_vel.set_ylabel("velocity")
                axs_vel.set_xlabel("episode_step")

                additional_data = {
                    "eval/num_wiped_markers": np.amax([df["num_wiped_markers"].max() for df in df_lst]),
                    "eval/force_dist": wandb.Histogram([df.loc[step, "force"] for df in df_lst for step in df.index]),
                    "eval/force_trace": fig_force,
                    "eval/vel_trace": fig_vel,
                }
                eval_data = {**eval_data, **additional_data}
            
            self.wandb_logger.log(eval_data)

            # log video of sim robot, if recorded
            if bool(video_frames):
                t1 = threading.Thread(
                    target=self.wandb_logger.log_video, 
                    kwargs={"data":video_frames, "commit":False, "step":self.update_steps}
                )
                t1.start()
                time.sleep(1e-2)
                t1.join()


        elif type == "pause-training":
            self.training_paused = True

        elif type == "continue-training":
            self.training_paused = False
        
        elif type == "stop-training":
            self.finished_run = True

        elif type == "store-best-model":
            best_model_dir = os.path.join(self.learner_config.checkpoint_path, "best_model")
            if not os.path.exists(best_model_dir):
                os.makedirs(best_model_dir)
            best_model_path = os.path.join(best_model_dir, "best_model.pt")

            torch.save(payload, best_model_path)

    def learn(self):
        """
        Main training loop for the asynchronous learner.
        
        This method implements the main training loop for the asynchronous learner. It waits for the actor to collect learning_starts
        number of samples before starting training. It then iteratively samples a batch of transitions from the replay buffer and trains 
        the agent on them for one gradient step. At fixed intervals of steps, the current network weights are sent to the actor, training
        metrics are logged, and the model is checkpointed. The training loop runs for max_steps number of steps.
        """
        
        # Wait to fill up replay buffer (corresponds to learning start in res_rl_uncertainty)
        pbar = tqdm.tqdm(total=self.learner_config.learning_starts,
                         initial=len(self.replay_buffer),
                         desc="Filling up the replay buffer.",
                         position=0,
                         leave=True)
        
        while len(self.replay_buffer) < self.learner_config.learning_starts:
            pbar.update(len(self.replay_buffer) - pbar.n)
            time.sleep(1)
        pbar.update(len(self.replay_buffer) - pbar.n)
        pbar.close()

        # Send initial network to actor (RL policy)
        self.trainer_server.publish_network(self.agent.agent.state_dict())

        timer = Timer()

        updates_left = self.updates_left
        self.updates_left = 0

        for step in tqdm.tqdm(range(self.learner_config.max_steps), dynamic_ncols=True, desc="learner"):

            # Wait for actor during evaluation
            with timer.context("trainer_waiting"):
                while_counter = 0
                while self.training_paused:
                    time.sleep(0.1)
                    while_counter += 1
                    if while_counter >= 1000:
                        # when we were in the while loop for 1000 times -> exit complete trainer loop
                        break
                else:
                    if updates_left == 0:
                        updates_left = self.updates_left
                        self.updates_left = 0
                    else:
                        updates_left -= 1
            
            with timer.context("sample_replay_buffer"):
                batch = self.replay_buffer.sample(self.batch_size)

            with timer.context("network_step"):
                train_metrics = self.agent.train(batch)

            # Current actor callback directly updates the network after each episodes (from actor) -> Maybe inbetween rollouts??
            if step > 0 and updates_left == 0:
                self.trainer_server.publish_network(self.agent.agent.state_dict())
                
            # log training stats and time stats at interval
            if step % self.learner_config.log_frequency == 0:
                self.wandb_logger.log(train_metrics, commit=False)
                self.wandb_logger.log({"buffer/position_percentage": self.replay_buffer.get_buffer_richness})
                self.wandb_logger.log({"steps": {"learner_step": step}}, commit=False)
                self.wandb_logger.log({"steps": {"episodes_received": self.episodes_received, "episodes_trained": self.episodes_trained}}, commit=False)
                self.wandb_logger.log({"timer": timer.get_average_times()})

            # checkpoint model
            if step % self.learner_config.checkpoint_frequency == 0:
                self._save_checkpoint(overwrite=self.learner_config.overwrite)
            
            self.update_steps += 1

            # check if training updates can be performed or if we have to wait for the actor
            if updates_left == 0:
                self.episodes_trained += 1
                # check whether actor finished or not
                if self.updates_left > 0:
                    # if self.updates_left is above 0 it means that actor is already done
                    updates_left = self.updates_left
                    self.updates_left = 0
                else:
                    # if no stats_update of actor came yet, set training to paused
                    self.training_paused = True
            else:
                pass

        print("Stopping learner client")
        self.trainer_server.stop()
        wandb.finish()
        print("Learner loop finished")

    def _save_checkpoint(self, overwrite: bool = True):
        """
        Save a checkpoint of the current model and replay buffer.
        
        This method saves a checkpoint of the current model and replay buffer. The checkpoint is saved in the directory 
        checkpoint_path/checkpoint_{update_steps}/ where update_steps is the current number of training steps taken and checkpoint_path
        is specified in the learner configuration. In this directory it will store the model state (weights, optimizer states, etc.) under
        agent.pt and the replay buffer under rb.pkl.
        """
        experiment_path = Path(self.learner_config.checkpoint_path)

        if overwrite:
            prev_checkpoints = [dir for dir in os.listdir(experiment_path) if dir.startswith("check")]
            for dir in prev_checkpoints:
                shutil.rmtree(os.path.join(experiment_path, dir))
            
        checkpoint_path = os.path.join(experiment_path, f"checkpoint_{self.update_steps}")
        
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        model_path = os.path.join(checkpoint_path, "agent.pt")
        rb_path = os.path.join(checkpoint_path, "rb")

        # save agent
        self.agent.save(model_path)

        # save replay buffer
        save_to_pkl(rb_path, self.replay_buffer)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a checkpoint of the model and replay buffer.

        This method loads a checkpoint of the model and replay buffer from the specified checkpoint_path and updates the agent and replay buffer
        with the loaded checkpoint. The checkpoint must be stored in the directory checkpoint_path/ where the model state is stored in agent.pt and
        the replay buffer is stored in rb.pkl.

        Args:
            checkpoint_path: The path to the checkpoint directory.
        """
        model_path = os.path.join(checkpoint_path, "agent.pt")
        rb_path = os.path.join(checkpoint_path, "rb")

        # load agent
        self.agent.load(model_path)

        # load replay buffer
        self.replay_buffer = load_from_pkl(rb_path)
        assert isinstance(self.replay_buffer, ReplayBufferDataStore), "The replay buffer must inherit from ReplayBuffer class"
        # Update saved replay buffer device to match current setting
        self.replay_buffer.device = self.learner_config.device
