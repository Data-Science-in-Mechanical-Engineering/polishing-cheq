# This code was taken and adapted from CleanRL
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py

import torch
from torch import nn
import torch.nn.functional as F
import gymnasium as gym
from agents.rl.policies import SACPolicy
from agents.rl.rl_base import RLBase
from typing import Type, List, Literal, Dict


class SAC(RLBase):
    """
    Class for training a SAC agent.
    Implements functionality for storing transitions in replay buffer and updating the weights of the SAC networks.

    Params:
      env: The environment to learn from. Must be a gym environment.
      device: The device the model is trained on.
      gamma: The discount factor.
      polyak_factor: The soft update coefficient of the target networks.
      policy_lr: The learning rate of the actor network
      q_lr: The learning rate of the critic networks
      policy_frequency: The frequency of updating the actor network
      target_network_frequency: The frequency of updating the target networks
      alpha: Entropy regularization coefficient if not tuned automatically
      autotune: Automatic tuning of the entropy regularization coefficient
      critic_hidden_dims: List of number of units per critic hidden layer. The length of the list corresponds to
                          the number of hidden layers of the critic networks.
      actor_hidden_dims: List of number of units per actor hidden layer. The length of the list corresponds to
                         the number of hidden layers of the actor network.
      critic_activation_fn: Activation function used in the critic networks.
      actor_activation_fn: Activation function used in the actor network.
      action_noise: The type of action noise to use for exploration. Must be either 'white' or 'pink'
    """

    def __init__(
            self,
            env: Type[gym.Env],
            device: torch.device = 'cpu',
            gamma: float = 0.99,
            polyak_factor: float = 0.005,
            policy_lr: float = 3e-4,
            q_lr: float = 1e-3,
            policy_frequency: int = 2,
            target_network_frequency: int = 1,
            alpha: float = 0.2,
            autotune: bool = True,
            critic_hidden_dims: List[int] = None,
            actor_hidden_dims: List[int] = None,
            critic_activation_fn: Type[nn.Module] = nn.ReLU,
            actor_activation_fn: Type[nn.Module] = nn.ReLU,
            action_noise: Literal['white', 'pink'] = 'white',
            layer_norm_q: bool = False,
            layer_norm_a: bool = False,
    ) -> None:

        # set default dimensions
        if actor_hidden_dims is None:
            actor_hidden_dims = [256, 256]
        if critic_hidden_dims is None:
            critic_hidden_dims = [256, 256]

        # store environment
        self.env = env

        # store required hyperparameter
        self.gamma = gamma
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.autotune = autotune
        self.polyak_factor = polyak_factor
        self.device = device

        # initialize number gradient steps with zero
        self.num_gradient_steps = 0

        # Training metrics dict
        self.step_metrics = dict()

        # initialize agent
        self.agent = SACPolicy(env=env, 
                               critic_hidden_dims=critic_hidden_dims, 
                               actor_hidden_dims=actor_hidden_dims,
                               device=self.device, 
                               critic_activation_fn=critic_activation_fn, 
                               actor_activation_fn=actor_activation_fn,
                               action_noise=action_noise,
                               layer_norm_q=layer_norm_q,
                               layer_norm_a=layer_norm_a).to(device)

        # initialize optimizers by concatenating q nets
        self.q_optimizer = torch.optim.Adam(list(self.agent.qf1.parameters()) + list(self.agent.qf2.parameters()),
                                            lr=q_lr)
        self.actor_optimizer = torch.optim.Adam(self.agent.actor.parameters(), lr=policy_lr)

        # initialize automatic entropy tuning
        if self.autotune:
            self.target_entropy = - torch.prod(torch.tensor(env.action_space.shape).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = torch.optim.Adam([self.log_alpha], lr=q_lr)
        else:
            self.alpha = alpha

        # the number of trainer updates performed
        self.parameters_updated = 0

    def update_target_networks(self) -> None:
        """
        Update the target networks by moving in the direction of the policy networks, weighted by the polyak averaging factor.
        Theta_Q_target = (1 - polyak_factor) * Theta_Q_target + polyak_factor * Theta_Q
        Takes the polyak_factor from the instance
        """
        with torch.no_grad():
            # zip does not raise an exception if length of parameters does not match.
            for param, target_param in zip(self.agent.qf1.parameters(), self.agent.qf1_target.parameters()):
                target_param.data.copy_(self.polyak_factor * param.data + (1 - self.polyak_factor) * target_param.data)
            for param, target_param in zip(self.agent.qf2.parameters(), self.agent.qf2_target.parameters()):
                target_param.data.copy_(self.polyak_factor * param.data + (1 - self.polyak_factor) * target_param.data)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False, learning_started: bool = True) -> torch.Tensor:
        """
        Gets the action from the SAC agent based on the provided observation.
        
        This method returns the SAC agent's action based on the provided observation. The agent outputs different
        actions depending on whether learning has started (random action sampled from environment if not) and whether the 
        action should be deterministic (no action noise).

        Args:
            obs: The current observation.
            deterministic: Whether the action should be deterministic.
            learning_started: Whether learning of the RL agent has started.

        Returns:
            The action to be taken.
        """
        if learning_started:
            action = self.agent.get_action(observation=obs.to(self.device), 
                                           deterministic=deterministic)
        else:
            action = torch.tensor(self.agent.actor.action_noise.sample(), device=self.device)

        return action

    def train(self, batch) -> Dict:
        """
        Trains the SAC agent's weights based on a batch of data, taking one gradient step.

        Args:
            batch: A batch of data to train on consisting of transitions (s, a, r, s', done).

        Returns:
            Dictionary of step metrics and performance.
        """

        # increase number of taken gradient steps
        self.num_gradient_steps += 1

        # initialize step metrics dict
        step_metrics = dict()

        with torch.no_grad():
            # get log_policy (for entropy), sample next action from pi and compute q_targets given the action
            next_state_actions, next_state_log_pi = self.agent.get_action_and_log_prob(batch.next_observations)
            qf1_next_target = self.agent.qf1_target(batch.next_observations, next_state_actions)
            qf2_next_target = self.agent.qf2_target(batch.next_observations, next_state_actions)

            # determine minimum of q_targets and compute y (TD-target)
            # further include logic for already done trajectories
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = batch.rewards.flatten() + (1 - batch.dones.flatten()) * self.gamma * min_qf_next_target.view(-1)

        # compute q values for given batch data
        qf1_a_values = self.agent.qf1(batch.observations, batch.actions).view(-1)
        qf2_a_values = self.agent.qf2(batch.observations, batch.actions).view(-1)
        # compute the mean squared error for both q nets (already divided by sum).
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss  # since optimizer is concatenated q nets.

        # perform optimization step of the model
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        # add q-metrics to step metrics
        step_metrics["agent_train/qf1_values"] = qf1_a_values.mean().item()
        step_metrics["agent_train/qf2_values"] = qf2_a_values.mean().item()
        step_metrics["agent_train/qf1_loss"] = qf1_loss.item()
        step_metrics["agent_train/qf2_loss"] = qf2_loss.item()
        step_metrics["agent_train/q_loss"] = qf_loss.item()

        # update of phi (policy net)
        if self.num_gradient_steps % self.policy_frequency == 0:  # TD3 Delayed update support
            for _ in range(self.policy_frequency):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                # compute log_policy (for entropy), sample actions and compute q_targets given s and a
                pi, log_pi = self.agent.get_action_and_log_prob(batch.observations)
                qf1_pi = self.agent.qf1(batch.observations, pi)
                qf2_pi = self.agent.qf2(batch.observations, pi)

                # determine minimum and compute average loss for optimization
                min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)  # !!! I changed this here!
                actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                # perform optimization step of model
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # update entropy regularization term alpha
                if self.autotune:  
                    # compute loss for optimization
                    with torch.no_grad():
                        _, log_pi = self.agent.get_action_and_log_prob(batch.observations)
                    alpha_loss = (-self.log_alpha * (log_pi + self.target_entropy)).mean()  # !!! I changed this here!

                    # perform optimization
                    self.a_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.a_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()

                    # add entropy tuning loss to step metrics
                    step_metrics["agent_train/entropy_coefficient_loss"] = alpha_loss.item()

            # add actor metrics to step metrics
            step_metrics["agent_train/actor_loss"] = actor_loss.item()
            step_metrics["agent_train/actor_entropy"] = -log_pi.detach().mean().item()
            step_metrics["agent_train/entropy_coefficient"] = self.alpha
            step_metrics["agent_train/target_entropy"] = self.target_entropy

        # update the target networks
        if self.num_gradient_steps % self.target_network_frequency == 0:
            self.update_target_networks()

        # update step metrics to not loose actor metrics if policy update is not performed
        self.step_metrics.update(step_metrics)

        return self.step_metrics

    def update_params(self, params: dict) -> None:
        """
        Updates the SAC agent's network parameters, with the passed parameters.

        This method is set as a callback in the asynchronous actor cycle to iteratively update the actor's weights with the
        weights of the learner's actor network given as a pytorch state_dict.

        It further updates a number called parameters_updated to track the updates from the learner.

        Args:
            params (dict): A dictionary containing the new network parameters to be updated in the form of
                           torch's get_state_dict() method.
        """
        self.agent.load_state_dict(state_dict=params)
        # for checking purposes
        self.parameters_updated += 1
    
    def save(self, save_path):
        """
        Saves the SAC agent's weights and optimizer states to the specified path.

        Args:
            save_path: The path to save the SAC agent to.
        """
        # save the agent and optimizer states
        agent_save_dict = {'agent_state_dict': self.agent.state_dict(),
                           'actor_optimizer_state': self.actor_optimizer.state_dict(),
                           'critic_optimizer_state': self.q_optimizer.state_dict()}

        if self.autotune:
            agent_save_dict['alpha_optimizer_state'] = self.a_optimizer.state_dict()

        torch.save(obj=agent_save_dict, f=save_path)

    def load(self, load_path):
        """
        Loads the SAC agent's weights and optimizer states from the specified path.

        Args:
            load_path: The path to load the SAC agent from.
        """
        # load the agent and optimizer states
        agent_states = torch.load(f=load_path)
        self.agent.load_state_dict(agent_states['agent_state_dict'])
        self.actor_optimizer.load_state_dict(agent_states['actor_optimizer_state'])
        self.q_optimizer.load_state_dict(agent_states['critic_optimizer_state'])

        if self.autotune:
            self.a_optimizer.load_state_dict(agent_states['alpha_optimizer_state'])

        self.agent.to(device=self.device)
