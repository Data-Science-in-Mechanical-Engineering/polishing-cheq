# This code was taken and adapted from CleanRL 
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py

import torch
from torch import nn
import numpy as np
import gymnasium as gym
from pink.cnrl import ColoredNoiseProcess
from typing import Type, List, Literal, Tuple, Union
from utils.network_utils import create_mlp


class QNetwork(nn.Module):
    """
    General MLP Q-function implementation.

    This class implements a general MLP implementation of the state-action value function Q(s,a).

    Attributes:
        q_net: The Q-network model
    """

    def __init__(
            self,
            env: Type[gym.Env],
            hidden_sizes: List[int],
            activation_fn: Type[nn.Module] = nn.ReLU,
            layer_norm: bool = False,
    ):
        """
        Initializes the Q-function MLP.
        
        Args:
            env: The environment in which the rl-agent operates.
            hidden_sizes: List of number of units per hidden layer. The length of the list corresponds to the number of hidden 
                          layers.
            activation_fn: The activation function to use after each layer.
        """

        super().__init__()
        mlp_layers = create_mlp(input_dim=np.prod(env.unwrapped.observation_space.shape) + np.prod(env.unwrapped.action_space.shape), 
                                output_dim=1, 
                                hidden_dims=hidden_sizes,
                                activation_fn=activation_fn,
                                layer_norm=layer_norm)

        self.q_net = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.float32:
        """
        Computes the Q-value for the given state-action pair.
        
        Args:
            x: The state input.
            a: The action input.
            
        Returns:
            The Q-value for the given state-action pair.
        """
        inp = torch.cat([x, a], -1)
        return self.q_net(inp)


class SACActor(nn.Module):
    """
    Actor Network for Soft Actor-Critic.

    This class implements the policy network for the Soft Actor-Critic algorithm. The policy network learns a stochastic
    policy mapping observations to a distribution over actions.

    Attributes:
        sequence_length: The maximum length of an episode. Is required for pink action noise.
        action_dim: The dimension of the action space.
        latent_pi: The latent policy network.
        mu: The network for the mean of the action distribution.
        log_std: The network for the log standard deviation of the action distribution.
        action_noise: The distribution to sample action noise from.
    """

    def __init__(
            self,
            env: Type[gym.Env],
            hidden_sizes: List[int],
            device: torch.device = 'cpu',
            activation_fn: Type[nn.Module] = nn.ReLU,
            action_noise: Literal['white', 'pink'] = 'white',
            layer_norm: bool = False,
    ):
        """
        Initializes the SAC Actor Network.
        
        Args:
            env: The environment in which the rl-agent operates.
            hidden_sizes: List of number of units per hidden layer. The length of the list corresponds to the number of hidden 
                          layers.
            activation_fn: The activation function to use after each layer.
            action_noise: The type of action noise to use for exploration. Must be either 'white' or 'pink'.
        """

        super().__init__()

        self.device = device

        self.LOG_STD_MIN = -5
        self.LOG_STD_MAX = 2

        self.sequence_length = env.unwrapped.horizon
        self.action_dim = np.prod(env.unwrapped.action_space.shape)
        observation_dim = np.prod(env.unwrapped.observation_space.shape)
        last_layer_dim = hidden_sizes[-1] if len(hidden_sizes) > 0 else observation_dim

        latent_pi_net = create_mlp(observation_dim, -1, hidden_sizes, activation_fn, layer_norm=layer_norm)  # -1 as args[1] since we want no output_dim
        self.latent_pi = nn.Sequential(*latent_pi_net)
        self.mu = nn.Linear(last_layer_dim, self.action_dim)
        self.log_std = nn.Linear(last_layer_dim, self.action_dim)

        assert action_noise in ['white', 'pink'], "Invalid action noise. Must be either 'white' or 'pink'."
        self.action_noise = self.initialize_action_noise(action_noise)

        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(
            self, 
            observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This method computes the mean and log standard deviation of the action distribution for the given observation
        using the mean and log_std networks. The log standard deviation is squashed to be within the range [-5, 2], as
        proposed here https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py.
        
        Args:
            observation: The input observation.
            
        Returns:
            The mean and log standard deviation of the action distribution.
        """
        latent_pi = self.latent_pi(observation)
        mean_actions = self.mu(latent_pi)
        log_std = self.log_std(latent_pi)
        # Squash log_std to be within [LOG_STD_MIN, LOG_STD_MAX]
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats (sac repo)

        return mean_actions, log_std

    def get_action(
            self, 
            observation: torch.Tensor, 
            deterministic: bool = False
    ) -> torch.Tensor:
        """
        Samples an action from the policy distribution for the given observation.
        
        This method samples as action from the policy distribution for the given observation in the case of stochastic actions
        (deterministic=False) while it returns the 'best' action in the case of deterministic actions (deterministic=True). The
        returned actions are scaled to the action space of the environment.
        
        Args:
            observation: The observation of the environment.
            deterministic: Whether to sample a stochastic or deterministic action.

        Returns:
            The sampled action.
        """
        mean, log_std = self(observation.to(torch.float))
        std = log_std.exp()

        if deterministic:
            x_t = mean
        else:
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample().to(self.device).to(torch.float)  # Reparameterization trick of torch backprop

        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        return action

    def get_action_and_log_prob(
            self, 
            observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples an action from the policy distribution for the given observation and computes the log probability of the action.

        Args:
            observation: The observation of the environment.

        Returns:
            The sampled action and the log probability of the action.
        """
        mean, log_std = self(observation)
        std = log_std.exp()
        action_dist = torch.distributions.Normal(mean, std)
        x_t = action_dist.rsample().to(self.device).to(torch.float)  # reparametrization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = action_dist.log_prob(x_t).to(self.device).to(torch.float)

        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob

    def initialize_action_noise(self, action_noise_type: str) -> Union[torch.distributions.Normal, ColoredNoiseProcess]:
        """
        Initializes the action noise distribution.

        This method initializes the action noise distribution based on the provided action noise type. This can be either the typical
        white noise distribution N(0,I) or a pink noise distribution where samples are correlated over time, as proposed in 
        https://openreview.net/forum?id=hQ9V5QN27eS

        Args:
            action_noise_type: The type of action noise to use for exploration. Must be either 'white' or 'pink'.

        Returns:
            The action noise distribution to sample from.
        """
        if action_noise_type == 'white':
            return torch.distributions.Normal(torch.zeros(self.action_dim), torch.ones(self.action_dim))

        elif action_noise_type == 'pink':
            assert self.sequence_length is not None, "Must provide a sequence length with pink action noise"
            return ColoredNoiseProcess(beta=1, size=(self.action_dim, self.sequence_length), scale=0.3887)  # refers to 99% between [-1,1]

    def reset_action_noise(self):
        if isinstance(self.action_noise, ColoredNoiseProcess):
            self.action_noise.reset()


class SACPolicy(nn.Module):
    """
    Policy Class that combines Actor and Critic networks for SAC.

    Attributes:
        qf1: The first Q-network.
        qf2: The second Q-network.
        qf1_target: The target network for the first Q-network.
        qf2_target: The target network for the second Q-network.
        actor: The actor network.
    """

    def __init__(
            self,
            env: Type[gym.Env],
            critic_hidden_dims: List[int],
            actor_hidden_dims: List[int],
            device: torch.device = "cpu",
            critic_activation_fn: Type[nn.Module] = nn.ReLU,
            actor_activation_fn: Type[nn.Module] = nn.ReLU,
            action_noise: Literal['white', 'pink'] = 'white',
            layer_norm_q: bool = False,
            layer_norm_a: bool = False,
        ):
        """
        Initializes the SAC Policy.
        
        Args:
            env: The environment in which the rl-agent operates.
            critic_hidden_dims: List of number of units per critic hidden layer. The length of the list corresponds to
                                the number of hidden layers of the critic networks.
            actor_hidden_dims: List of number of units per actor hidden layer. The length of the list corresponds to
                                the number of hidden layers of the actor network.
            critic_activation_fn: Activation function used in the critic networks.
            actor_activation_fn: Activation function used in the actor network.
            action_noise: The type of action noise to use for exploration. Must be either 'white' or 'pink'
        """
        super().__init__()

        self.device = device

        # Initialize Actor and Critic networks
        self.qf1 = QNetwork(env, critic_hidden_dims, critic_activation_fn, layer_norm=layer_norm_q)
        self.qf2 = QNetwork(env, critic_hidden_dims, critic_activation_fn, layer_norm=layer_norm_q)
        self.qf1_target = QNetwork(env, critic_hidden_dims, critic_activation_fn, layer_norm=layer_norm_q)
        self.qf2_target = QNetwork(env, critic_hidden_dims, critic_activation_fn, layer_norm=layer_norm_q)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        self.actor = SACActor(env, actor_hidden_dims, self.device, actor_activation_fn, action_noise, layer_norm=layer_norm_a)

    def get_action(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self.actor.get_action(observation, deterministic)

    def get_action_and_log_prob(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.actor.get_action_and_log_prob(observation)

    def reset_action_noise(self):
        self.actor.reset_action_noise()


class EnsembleSACPolicy(nn.Module):
    """
    Policy Class that combines Actor and Critic networks for the CHEQ SAC-ensemble.

    Attributes:
        ensemble: The critic ensemble initialized with the esemble-size.
        ensemble_targets: The critic ensemble targets initialized with the esemble-size and the states from the ensemble.
        actor: The actor network.
    """
    def __init__(
            self,
            env: Type[gym.Env],
            ensemble_size: int,
            critic_hidden_dims: List[int],
            actor_hidden_dims: List[int],
            device: torch.device = 'cpu',
            critic_activation_fn: Type[nn.Module] = nn.ReLU,
            actor_activation_fn: Type[nn.Module] = nn.ReLU,
            action_noise: Literal['white', 'pink'] = 'white',
            layer_norm_q: bool = False,
            layer_norm_a: bool = False,
    ) -> None:
        
        super().__init__()
        
        self.ensemble_size = ensemble_size
        self.device = device

        # initialize Q-net ensemble and actor
        self.ensemble = nn.ModuleList([QNetwork(env=env, hidden_sizes=critic_hidden_dims, activation_fn=critic_activation_fn, layer_norm=layer_norm_q) for _ in range(self.ensemble_size)])
        self.ensemble_targets = nn.ModuleList([QNetwork(env=env, hidden_sizes=critic_hidden_dims, activation_fn=critic_activation_fn, layer_norm=layer_norm_q) for _ in range(self.ensemble_size)])

        for j in range(self.ensemble_size):
            self.ensemble_targets[j].load_state_dict(self.ensemble[j].state_dict())

        self.actor = SACActor(
            env=env, 
            hidden_sizes=actor_hidden_dims, 
            device=self.device, 
            activation_fn=actor_activation_fn, 
            action_noise=action_noise,
            layer_norm=layer_norm_a,
        )

    def get_action(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Given a specific observation, defines the action to take.
        Further includes a deterministic boolean which indicates if the action should be deterministic or stochastic.

        Args:
            observation (torch.Tensor): The state.
            deterministic (bool, optional): Defining the type of policy. Defaults to False.

        Returns:
            torch.Tensor: The next action.
        """
        return self.actor.get_action(observation=observation, deterministic=deterministic)

    def get_action_and_log_prob(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples an action from the policy distribution for the given observation 
        and computes the log probability of the action.

        Args:
            observation (torch.Tensor): The state.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple consisting of action and log_prob.
        """
        return self.actor.get_action_and_log_prob(observation=observation)

    def reset_action_noise(self) -> None:
        """
        Resets the action noise buffer to empty.
        """
        self.actor.reset_action_noise()

    def get_q_value(self, state: torch.Tensor, action: torch.Tensor, q_net: int, target: bool = False) -> torch.Tensor:
        """
        Receive a forward pass of the current q-network specified with q_net,
        given a state and an action.

        Args:
            state (torch.Tensor): The state to be evaluated
            action (torch.Tensor): The action taken in the state.
            q_net (int): Which of the network of the q-ensemble should be taken?
            target (bool): Should the ensemble or the target-ensemble be taken?

        Returns:
            torch.Tensor: The q-value of the determined network.
        """
        assert q_net <= self.ensemble_size, "The requested q_net is above the specified q-ensembe size."

        if target:
            return self.ensemble_targets[q_net](x=state.to(torch.float32), a=action.to(torch.float32))
        else:
            return self.ensemble[q_net](x=state.to(torch.float32), a=action.to(torch.float32))
