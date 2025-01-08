from agents.hybrid.hybrid_base import HybridBase
from agents.rl.rl_base import RLBase
from agents.nominal.nominal_base import NominalBase
from typing import Type

import torch
import numpy as np


class HybridAgent(HybridBase):
    """
    Basic hybrid agent with constant mixing parameter.

    Basic hybrid agent that combines actions of the RL-agent and nominal-agent with a constant mixing parameter eta.
    The merged action results from: u_merged = eta*u_rl + (1-eta)*u_nominal. Therefore, a value of eta=1 corresponds 
    to pure RL-action while a value of eta=0 results in pure nominal action.

    Attributes:
        rl_agent: The reinforcement learning agent used by the hybrid agent.
        nominal_agent: The nominal agent used by the hybrid agent.
        eta: The constant mixing parameter. Must take values in [0, 1].
    """
    
    def __init__(
            self,
            rl_agent: Type[RLBase],
            nominal_agent: Type[NominalBase],
            eta: float
    ):
        """
        Initializes the hybrid agent based on the provided reinforcement learning and nominal agents and the mixing parameter eta.
        
        Args:
            rl_agent: The reinforcement learning agent used by the hybrid agent.
            nominal_agent: The nominal agent used by the hybrid agent.
            eta: The constant mixing parameter. Must take values in [0, 1].
        """
        
        super().__init__(rl_agent, nominal_agent)

        assert 0 <= eta <= 1, 'The mixing parameter eta must take values in [0,1]'
        self.eta = eta
        
    def get_action(
            self,
            obs: torch.Tensor, 
            deterministic: bool = False, 
            learning_started: bool = True):
        """
        Returns the hybrid and rl-action based on the provided observation.

        This method gets the rl-action from the rl-agent and the nominal-action from the nominal-agent and computes the mixed hybrid
        action according to mixed_action = eta*rl_action + (1-eta)*nominal_action.

        Args:
            obs: The current observation.
            deterministic: Whether the rl agent's action should be deterministic (no action noise).
            learning_started: Whether learning of the rl agent has started (otherwise action randomly sampled from action space).

        Returns:
            A tuple of form (mixed_action, rl_action)
        """
        
        rl_action = self.rl_agent.get_action(obs=obs, deterministic=deterministic, learning_started=learning_started)  # tensor at device
        nominal_action = self.nominal_agent.get_action(obs=obs)  # tensor at device

        mixed_action = self.eta * rl_action + (1 - self.eta) * nominal_action
        
        return mixed_action, rl_action
    
    def get_mixing_parameter(self):
        """
        Returns the mixing parameter.
        """
        return self.eta
    
    def update_mixing_parameter(self, step: int):
        """
        Updates the mixing parameter. Here the mixing parameter is constant and does not change.
        """
        pass

    def reset_mixing_parameter(self):
        """
        Resets the mixing parameter. Here the mixing parameter is constant and does not change.
        """
        pass

    @property
    def get_uncertainty(self):
        """
        Since we do not have any uncertainty in the normal Hybrid Agent!
        """
        return None
