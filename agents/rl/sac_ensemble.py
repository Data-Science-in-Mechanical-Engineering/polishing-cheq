# @author: Lukas Jäschke
# @source: https://git.rwth-aachen.de/dsme-projects/grinding_robot/res_rl_uncertainty

from agents.rl.rl_base import RLBase
from agents.rl.sac import SAC
from agents.rl.policies import EnsembleSACPolicy
from typing import Dict, List, Literal, Type, Optional, Union, Tuple
from utils.data_utils import BernoulliMaskReplayBufferSamples
from stable_baselines3.common.type_aliases import ReplayBufferSamples
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np


class SACEnsemble(RLBase):
    """
    Class of the contextualized Ensemble SAC agent.
    """

    def __init__(
            self,
            env: Type[gym.Env],
            ensemble_size: int,
            device: torch.device = 'cpu',
            gamma: float = 0.99,
            polyak_factor: float = 0.005,
            policy_lr: float = 3e-4,
            q_lr: float = 1e-3,
            policy_frequency: int = 2,
            pi_update_avg_q: bool = False,
            target_network_frequency: int = 1,
            alpha: float = 0.2,
            autotune: bool = True,
            critic_hidden_dims: Optional[List[int]] = None,
            actor_hidden_dims: Optional[List[int]] = None,
            critic_activation_fn: Type[nn.Module] = nn.ReLU,
            actor_activation_fn: Type[nn.Module] = nn.ReLU,
            action_noise: Literal['white', 'pink'] = 'white',
            layer_norm_q: bool = False,
            layer_norm_a: bool = False,
    ) -> None:
        """
        Initialize the EnsembleSAC agent and manage the training phase.

        Args:
            env (Type[gym.Env]): The environment to use.
            ensemble_size (int): The size of the ensemble.
            device (torch.device, optional): Which device to choose for torch operations. Defaults to 'cpu'.
            gamma (float, optional): Discount factor over future rewards. Defaults to 0.99.
            polyak_factor (float, optional): The smoothing coefficient for target q networks parameters. Defaults to 0.005.
            policy_lr (float, optional): The learning rate of the policy network optimizer. Defaults to 3e-4.
            q_lr (float, optional): The learning rate of the q network optimizer. Defaults to 1e-3.
            policy_frequency (int, optional): Frequency of policy updates (delayed!). Defaults to 2.
            pi_update_avg_q (bool, optional): Whether to use the average q value for the policy update. Defaults to False.
            target_network_frequency (int, optional): Frequency of updates for the target networks. Defaults to 1.
            alpha (float, optional): The entropy regularization coefficient. Defaults to 0.2.
            autotune (bool, optional): Automatic tuning of the entropy coefficient. Defaults to True.
            critic_hidden_dims (List[int], optional): List of number of units per hidden layer of the q-networks. Defaults to None.
            actor_hidden_dims (List[int], optional): List of number of units per hidden layer of the policy network. Defaults to None.
            critic_activation_fn (Type[nn.Module], optional): The activation function for the critic ensemble. Defaults to nn.ReLU.
            actor_activation_fn (Type[nn.Module], optional): The activation function for the policy. Defaults to nn.ReLU.
            action_noise (Literal['white', 'pink'], optional): The type of action noise to use for exploration. Defaults to 'white'.
        """
        
        # set default dimensions if not provided
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
        self.pi_update_avg_q = pi_update_avg_q
        self.ensemble_size = ensemble_size

        # initialize number gradient steps with zero
        self.num_gradient_steps = 0

        # Training metrics dict
        self.step_metrics = dict()

        # initialize agent
        self.agent = EnsembleSACPolicy(env=self.env,
                                       ensemble_size=self.ensemble_size, 
                                       critic_hidden_dims=critic_hidden_dims, 
                                       actor_hidden_dims=actor_hidden_dims,
                                       device=self.device, 
                                       critic_activation_fn=critic_activation_fn, 
                                       actor_activation_fn=actor_activation_fn,
                                       action_noise=action_noise,
                                       layer_norm_q=layer_norm_q,
                                       layer_norm_a=layer_norm_a).to(self.device)

        # initialize optimizers by concatenating q nets
        q_params = []
        for q_net in range(self.agent.ensemble_size):
            q_params += list(self.agent.ensemble[q_net].parameters())

        self.q_optimizer = torch.optim.Adam(params=q_params, 
                                            lr=q_lr)
        self.actor_optimizer = torch.optim.Adam(list(self.agent.actor.parameters()), 
                                                lr=policy_lr)

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

    def get_action(self, obs: torch.Tensor, deterministic: bool = False, learning_started: bool=True) -> torch.Tensor:
        """
        Gets the action from the RL agent based on the provided observation.
        
        This method should return the action based on the provided observation. The RL agent outputs different actions 
        depending on whether learning has started (random action sampled from environment if not) and whether the 
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

    def get_epistemic_uncertainty(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[float, float]:
        """
        Computes the epistemic uncertainty of the Q-ensemble 
        when given a specific action in an observation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]
            1. Tensor: The mean of the Q-ensemble.
            2. Tensor: The epistemic uncertainty of the Q-ensemble in the form of the standard deviation!
        """
        q_values = []

        # verify that both tensors are on the same device
        state = state.to(device=self.device) if state.get_device() == -1 else state
        action = action.to(device=action.device) if action.get_device() == -1 else action

        for q_net in range(self.ensemble_size):
            q_values.append(self.agent.get_q_value(state=state, action=action, q_net=q_net, target=False).unsqueeze(-1))

        q_values = torch.cat(q_values, dim=-1)
        mean = torch.mean(q_values, dim=-1)
        std = torch.std(q_values, dim=-1)

        return mean, std
        

    def train(self, batch: Union[ReplayBufferSamples, BernoulliMaskReplayBufferSamples]) -> Dict:
        """
        Trains the RL agent weights based on a batch of data.
        
        This method should update the RL agent's weights based on the provided batch of data taking one gradient step.
        The data should consist of a batch of transitions (s, a, r, s', done, (bern_mask)) sampled from the replay buffer.

        Note:
            In order for the UTD of the ensemble algorithm to work we have to give
            an extended batch with the dimensions of the update-to-data ratio.
        Args:
            batch: A batch of data to train on.

        Returns:
            A dictionary containing the training metrics to be logged (e.g. q-loss, actor loss etc.).
        """
        # loop over the specified update-to-data ratio
        # (the specified utd ratio is part of the dimensions of the batch)
        for update_step in range(batch.observations.shape[0]):  # could take any other attribute
            
            # initialize step metrics dict
            training_metrics = dict()
            
            # get current batch from big batch by __getitem__
            data = batch[update_step]

            # sample 2 random q-nets from the ensemble
            indices = np.random.choice(a=self.agent.ensemble_size, size=2, replace=False)

            with torch.no_grad():
                # get log_policy (for entropy), sample next action from pi and compute q_targets given the action
                next_state_actions, next_state_log_pi = self.agent.get_action_and_log_prob(observation=data.next_observations)
                q1_next_target = self.agent.get_q_value(state=data.next_observations, action=next_state_actions, q_net=indices[0], target=True)
                q2_next_target = self.agent.get_q_value(state=data.next_observations, action=next_state_actions, q_net=indices[1], target=True)

                min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.gamma * min_q_next_target.view(-1)

            q_a_values = [self.agent.get_q_value(state=data.observations, action=data.actions, q_net=q_net, target=False).view(-1) for q_net in range(self.ensemble_size)]
            
            # compute error loss for Q networks, use masking to determine inputs
            q_losses = [((q_a_values[q_net] - next_q_value)**2 * data.masks[:, q_net].view(-1)).sum() / torch.sum(data.masks[:, q_net]) for q_net in range(self.ensemble_size)]   
            q_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
            for loss in q_losses:
                q_loss = q_loss + loss

            # perform optimization for Q network
            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()

            # add q-metrics to train metrics
            training_metrics["agent_train/q_loss"] = q_loss.item()
            training_metrics["agent_train/next_q_values"] = next_q_value.mean() 

            # update the target networks
            if self.num_gradient_steps % self.target_network_frequency == 0:
                self.update_target_networks()
                
        # improve policy using entropy regularized policy update
        if self.num_gradient_steps % self.policy_frequency == 0:
            for _ in range(self.policy_frequency):  # compensate for the delay
                # compute log_policy (for entropy)
                pi, log_pi = self.agent.get_action_and_log_prob(observation=data.observations)
                
                if self.pi_update_avg_q:
                    # compute average q value of ensemble given the state and the action
                    q_vals = [self.agent.get_q_value(state=data.observations, action=pi, q_net=q_net, target=True) for q_net in range(self.ensemble_size)]
                    mean_q_pi = torch.sum(torch.stack(q_vals), dim=0) / self.ensemble_size
                    policy_loss = ((self.alpha * log_pi) - mean_q_pi).mean()
                else:
                    # determine min of sampled q-targets and compute average loss for optimization
                    q1_pi = self.agent.get_q_value(state=data.observations, action=pi, q_net=indices[0], target=True)
                    q2_pi = self.agent.get_q_value(state=data.observations, action=pi, q_net=indices[1], target=True)
                    min_q_pi = torch.min(q1_pi, q2_pi).view(-1)
                    policy_loss = ((self.alpha * log_pi) - min_q_pi).mean()

                # perform optimization for policy
                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                self.actor_optimizer.step()

                # update alpha parameter
                if self.autotune:
                    with torch.no_grad():
                        _, log_pi = self.agent.get_action_and_log_prob(observation=data.observations)
                    alpha_loss = (-self.log_alpha * (log_pi + self.target_entropy)).mean()

                    # tune entropy temperature
                    self.a_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.a_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()
                    
                    # add alpha metrics to train metrics
                    training_metrics["agent_train/entropy_coefficient_loss"] = alpha_loss.item()
                    
            
            # add actor metrics to train metrics
            training_metrics["agent_train/actor_loss"] = policy_loss.item()
            training_metrics["agent_train/actor_entropy"] = -log_pi.detach().mean().item()
            training_metrics["agent_train/entropy_coefficient"] = self.alpha
            training_metrics["agent_train/target_entropy"] = self.target_entropy
            
        # update step_metrics to not loose actor metrics when not performed
        self.step_metrics.update(training_metrics)
        
        # increase number of taken gradient steps
        self.num_gradient_steps += 1
        
        return self.step_metrics

    def update_params(self, params: dict) -> None:
        """
        Updates the RL agent's network parameters.

        This method should update the RL agent's network parameters based on the provided parameter dictionary.
        More specifically, this method is called within the asynchronous actor cycle to iteratively update the actor's
        weights with the weights of the learner's actor network given as a pytorch state_dict.

        It further updates a number called parameters_updated to track the updates from the learner.

        Args:
            params: A dictionary containing the new network parameters to be updated in the form of 
            torch's get_state_dict() method.
        """
        self.agent.load_state_dict(state_dict=params)
        # updates received
        self.parameters_updated += 1

    def update_target_networks(self) -> None:
        """
        Update the target q networks by moving in the direction of the normal q networks,
        weighted by the polyak averaging factor.
        
        Note:
            The polyak factor is taken from the initialization of the class.
        """
        with torch.no_grad():
            for q_net in range(self.ensemble_size):
                # zip does not raise an exception if lenght of parameters does not match
                for param, target_param in zip(self.agent.ensemble[q_net].parameters(), self.agent.ensemble_targets[q_net].parameters()):
                    target_param.data.mul_(1 - self.polyak_factor)
                    torch.add(target_param.data, param.data, alpha=self.polyak_factor, out=target_param.data)

    def save(self, save_path) -> None:
        """
        Saves the ensemble's weights and optimizer states to the specified path.

        This method should save the RL agent's network parameters as well as the optimizer states to the specified path.

        Args:
            save_path: The path to save the agents parameters to.
        """
        agents_save_dict = {
            "agent_state_dict": self.agent.state_dict(),
            "actor_optimizer_state": self.actor_optimizer.state_dict(),
            "critic_optimizer_state": self.q_optimizer.state_dict()
        }

        if self.autotune:
            agents_save_dict["log_alpha_value"] = self.log_alpha
            agents_save_dict["alpha_optimizer_state"] = self.a_optimizer.state_dict()
        
        torch.save(obj=agents_save_dict, f=save_path)

    def load(self, load_path) -> None:
        """
        Loads the ensemble agent's weights and optimizer states from the specified path.

        Args:
            load_path: The path to load the RL agent from.
        """
        # load the agent and optimizer states
        agent_states = torch.load(f=load_path)
        self.agent.load_state_dict(agent_states.get("agent_state_dict"))
        return
        self.actor_optimizer.load_state_dict(agent_states.get("actor_optimizer_state"))
        self.q_optimizer.load_state_dict(agent_states.get("critic_optimizer_state"))

        if self.autotune:
            self.log_alpha = agent_states.get("log_alpha_value")
            self.a_optimizer.load_state_dict(agent_states.get("alpha_optimizer_state"))
            self.alpha = self.log_alpha.exp().item()
        
        self.agent.to(device=self.device)

        