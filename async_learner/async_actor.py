# This code was inspired by SERL (https://serl-robot.github.io/)
# https://github.com/rail-berkeley/serl

from agentlace.data.data_store import QueuedDataStore
from agentlace.trainer import TrainerClient
from utils.timer_utils import Timer
from utils.launcher_utils import make_trainer_config
from utils.logging_utils import create_rollout_stats
from utils.plotting_utils import EvalObserver
from utils.data_utils import inject_weight_into_state
from agents.hybrid.hybrid_base import HybridBase
import tqdm
import sys
import os
import gymnasium as gym
import torch
import numpy as np
from typing import Dict, Type
import time


class AsyncActor:
    """
    Actor class for interacting with the environment asynchronously to training the agent.
    
    This class implements the actor side of the asynchronous RL setup, where training and interaction with the environment are decoupled 
    to enhance robustness and timing determinism in real-world learning tasks. It implements the base functionality for running the action
    loop with the environment while buffering the transitions and sending them to the learner for training at the end of the episode, as well
    as receiving the updated network from the learner. It also implements functionality for evaluating the agent on the environment.

    Attributes:
        agent: The hybrid agent to be trained.
        env: The environment to interact with.
        actor_config: Configuration for the actor.
        best_eval_return: The best return obtained during evaluation used for checkpointing the best model.
        data_store: The data store for buffering transitions before sending them to the learner.
        trainer_client: The client to send requests to the asynchronous learner.
    """

    def __init__(
            self, 
            agent: HybridBase, 
            env: Type[gym.Env], 
            actor_config: Dict,
            name_run: str,
    ):
        """
        Initializes the asynchronous actor.
        
        Args:
            agent: The hybrid agent to be trained.
            env: The environment to interact with.
            actor_config: Configuration for the actor.
        """
        
        self.agent = agent
        self.env = env
        self.actor_config = actor_config
        self.name_run = name_run

        # Check if buffer length is larger than environment horizon
        assert self.actor_config["buffer_length"] > self.env.unwrapped.horizon, "Buffer length must be larger than environment horizon to avoid loosing transitions!"
        
        self.best_eval_return = -np.inf

        self.data_store = QueuedDataStore(capacity=self.actor_config.buffer_length)

        self.trainer_client = TrainerClient(name="actor_env",
                                    server_ip=self.actor_config.learner_ip,
                                    config=make_trainer_config(),
                                    data_store=self.data_store,
                                    wait_for_server=True)

        self.trainer_client.recv_network_callback(self.agent.rl_agent.update_params)
        
        self.trainer_ready = False
        self.total_updates = 0

    def act(self):
        """
        This method performs the action loop between the actor and the environment it interacts with.
        
        This method interacts with the environment for max_steps number of steps, by iteratively sampling actions from the agent and
        executing them in the environment. The transitions are buffered and sent to the learner for training at the end of the episode.
        Furthermore, the episode's stats are sent to the learner for logging. If the evaluation frequency is reached, the agent is evaluated
        on the environment while the learner pauses training and the best model is checkpointed.
        """
        # Reset environment
        obs, _ = self.env.reset()
        done = False
        # Reset mixing parameter
        self.agent.reset_mixing_parameter()
        
        # training loop
        timer = Timer()
        episode_count = 0
        fails_count = 0
        last_evaluated_step = 0
        record_counts = 0

        # some lists for rollout logging purposes
        lst_mixing_parameters = []
        lst_uncertainties = []
        lst_rl_actions = []
        lst_hybrid_actions = []

        learning_started = False

        for step in tqdm.tqdm(range(-self.actor_config.learning_starts, self.actor_config.max_steps), dynamic_ncols=True):
            # get start time of step
            start_time = time.time()

            timer.tick("total_actor_step")

            with timer.context("receive_actions"):

                # Merge environment observation and mixing parameter
                # get s_t, lambda_t^rl
                if self.env.unwrapped.mixed_action_env:
                    mixing_parameter = self.agent.get_mixing_parameter()
                    obs = inject_weight_into_state(obs, mixing_parameter)
                else:
                    pass

                # get hybrid and rl action from hybrid agent
                # compute actions a_t^mix and a_t^rl
                with torch.no_grad():
                    hybrid_action, rl_action = self.agent.get_action(obs=obs, 
                                                                     deterministic=False,
                                                                     learning_started=learning_started)

                lst_hybrid_actions.append(hybrid_action.detach().cpu().numpy())
                lst_rl_actions.append(rl_action.detach().cpu().numpy())

            with timer.context("step_env"):

                next_obs, reward, terminated, truncated, info = self.env.step(hybrid_action)
                # Merge next environment observation and mixing parameter
                real_next_obs = inject_weight_into_state(next_obs, mixing_parameter) if self.env.unwrapped.mixed_action_env else next_obs.copy()
                info["mixing_parameter"] = mixing_parameter if self.env.unwrapped.mixed_action_env else -1
                # append for logging purposes
                lst_mixing_parameters.append(mixing_parameter)

            # Update mixing parameter and receive lambda_t+1^rl
            self.agent.update_mixing_parameter(step=step)
            # note that this appending is not 100% correct since we take the certainty of the next step here... (however very small effect for long episodes)
            lst_uncertainties.append(self.agent.get_uncertainty)

            # store transition in data buffer
            # note that this does not have an influence on using Bernoulli masking or not
            reward = np.asarray(reward, dtype=np.float32)
            done = terminated or truncated

            transition = dict(
                observations=obs.cpu().numpy(),
                actions=rl_action.cpu().numpy(),
                next_observations=real_next_obs.cpu().numpy(),
                rewards=reward,
                dones=truncated,  # only if episode ends unnaturally
            )
            self.data_store.insert(transition)

            # init next step
            obs = next_obs

            if done:
                # end timer for total step prematurely
                timer.tock("total_actor_step")

                # reset action space sampling for action_noise
                if learning_started: self.agent.rl_agent.agent.reset_action_noise() 

                # increment episode count
                episode_count += 1
                fails_count += 1 if truncated else 0

                info.update({"fails_count": fails_count})

                # Send stats to learner for logging
                stats = create_rollout_stats(
                    env=self.env,
                    lambda_lst=lst_mixing_parameters,
                    uncertainty_lst=lst_uncertainties,
                    hybrid_a_lst=lst_hybrid_actions,
                    rl_a_lst=lst_rl_actions,
                    info_dict=info,
                    learning_started=learning_started,
                    timer=timer,
                    step=step,
                )
                self.trainer_client.request("send-stats", stats)

                # Update learner with episodes data if trainer is ready for it
                with timer.context("actor_waiting"):
                    while_counter = 0
                    while not self.trainer_ready:
                        # sleep if trainer is not ready yet
                        time.sleep(0.1)
                        while_counter += 1
                        self.trainer_ready = self._check_trainer_availability() if learning_started else True
                        if while_counter >= 1000:
                            # when we were 1000 times in the while loop -> exit the complete actor
                            self.trainer_client.stop()
                            print("Actor loop finished")
                            sys.exit()
                    else:
                        # if ready: update replay buffer of learner and set to training_ready to false
                        self.trainer_client.update()
                        self.trainer_ready = False

                # perform bool getter here since we need buffer to be filled to start learning
                learning_started = step > 0

                # Evaluate agent
                if (self.actor_config.evaluation_frequency[1] == "steps" and (step-last_evaluated_step) >= self.actor_config.evaluation_frequency[0]) or (self.actor_config.evaluation_frequency[1] == "episodes" and episode_count % self.actor_config.evaluation_frequency[0] == 0):
                    # Update last evaluation step
                    last_evaluated_step = step
                    # Request pausing of training
                    time.sleep(1e-3)
                    self.trainer_client.request("pause-training", {})
                    # Evaluate agent
                    if self.actor_config.record_simulation[0] and record_counts % self.actor_config.record_simulation[1] == 0:
                        eval_return = self._eval_agent(
                            num_episodes=self.actor_config.evaluation_episodes, 
                            curr_actor_step=step,
                            record_video=self.actor_config.record_simulation[0]
                        )
                        record_counts += 1
                    else:
                        eval_return = self._eval_agent(
                            num_episodes=self.actor_config.evaluation_episodes,
                            curr_actor_step=step,
                            record_video=False
                        )
                    # Checkpoint best agent here
                    if eval_return > self.best_eval_return:
                        self.best_eval_return = eval_return
                        payload = {"agent_state_dict": self.agent.rl_agent.agent.state_dict()}
                        self.trainer_client.request("store-best-model", payload)
                        time.sleep(1)
                    # Resume training
                    self.trainer_client.request("continue-training", {})
                    
                # Reset env to start new episode
                obs, _ = self.env.reset()
                # Reset mixing parameter
                self.agent.reset_mixing_parameter()
                # Reset mixing parameter and uncertainty lists
                lst_mixing_parameters = []
                lst_uncertainties = []

            # wait in order to create constant frequency
            dt = time.time() - start_time
            time.sleep(max(0, (1/self.actor_config.control_freq) - dt))

            if "total_actor_step" in timer.start_times:
                # only end timer if it wasnt ended before due to done state
                timer.tock("total_actor_step")
            else:
                # if there is no running timer for total step, pass
                pass

        print("Stopping actor client")
        self.trainer_client.stop()
        print("Actor loop finished")

        
    def _eval_agent(self, num_episodes: int, curr_actor_step: int, record_video: bool = False) -> float:
        """
        Evaluate the agent on the environment for a given number of episodes and returns the average return of all evaluation runs.

        Here the agent is evaluated on the environment for a given number of episodes using the current policy. The resulting
        metrics during the rollout are buffered and sent to the learner for logging after every episode. At the end of all rollouts
        the average return accross the rollouts is returned.
        
        Args:
            num_episodes: The number of episodes to evaluate the agent for.
            
        Returns:
            The average return obtained by the agent over the given number of episodes.
        """
        # initialize reward array
        episode_rewards = [0] * num_episodes
        overall_stats = dict()
        eo = EvalObserver()

        for episode in range(num_episodes):
            # reset environment
            obs, _ = self.env.reset()
            done = False
            # reset mixing parameter
            self.agent.reset_mixing_parameter()
            # initialize dictionary for storing the episode statistics for logging
            episode_stats = dict()
            episode_step = 0
            best_run_reward = -np.inf
            frames = []

            while not done:
                # Merge environment observation and mixing parameter
                if self.env.unwrapped.mixed_action_env:
                    mixing_parameter = self.agent.get_mixing_parameter()
                    obs = inject_weight_into_state(obs, mixing_parameter)

                with torch.no_grad():
                    hybrid_action, _ = self.agent.get_action(obs=obs, deterministic=True, learning_started=True)
                next_obs, reward, terminated, truncated, info = self.env.step(hybrid_action)

                eo.register_step(episode_step, iteration=1)
                eo.register_state(obs, iteration=1)
                rescaled_hybrid_action = self.env.unwrapped.rescaling_func(hybrid_action.detach().cpu().numpy())
                eo.register_action(rescaled_hybrid_action, iteration=1, key="hybrid")

                eo.register_info(info, iteration=1)
                eo.register_reward(dict(self.env.unwrapped.reward_info), iteration=1)

                info["mixing_parameter"] = mixing_parameter if self.env.unwrapped.mixed_action_env else -1
                info["learning_started"] = curr_actor_step > 0

                if record_video:
                    out = self.env.render()
                    frames.append(out)

                # update mixing parameter based on the new action and state
                self.agent.update_mixing_parameter(step=curr_actor_step)

                # store step info
                episode_step += 1
                episode_stats[episode_step] = info
                episode_rewards[episode] += reward

                done = terminated | truncated
                obs = next_obs
                
                # when evaluation episode is done
                if done:
                    overall_stats[episode] = episode_stats
                    # check if it is the best run of the evaluations
                    if best_run_reward < episode_rewards[episode]:
                        best_run_reward = episode_rewards[episode]
                        final_frames = frames

            if episode_rewards[episode] > self.best_eval_return:
                if not os.path.exists(self.name_run):
                    os.makedirs(self.name_run)

                filename = f"{self.name_run}/evalobserver_{curr_actor_step}_return_{int(episode_rewards[episode])}.csv"
                eo.to_csv(filename)


        # only append best run of the evaluation as video
        if final_frames:
            overall_stats["video_frames"] = final_frames

        self.trainer_client.request("send-eval-metrics", overall_stats)

        return np.mean(episode_rewards)
      
    
    def _check_trainer_availability(self):
        """
        Check for trainer availability by verifying that the network got an update.
        If update was performed, update total_updates to new number and return True.

        Returns:
            bool: Trainer is available for update (True) or not (False).
        """
        if self.total_updates < self.agent.rl_agent.parameters_updated:
            self.total_updates = self.agent.rl_agent.parameters_updated
            return True
        else:
            return False