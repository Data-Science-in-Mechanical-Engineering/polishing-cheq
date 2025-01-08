from abc import ABC, abstractmethod
import numpy as np
from typing import Type, Tuple
from agents.rl.rl_base import RLBase
from agents.nominal.nominal_base import NominalBase


class HybridBase(ABC):
    """
    Abstract base class defining an interface for hybrid agents.

    Abstract base class defining an interface for hybrid agents consisting of a nominal agent and a reinforcement learning agent. All nominal
    agents must implement this class to be compatible with upstream training loops. 
    
    Attributes:
        rl_agent: The reinforcement learning agent used by the hybrid agent.
        nominal_agent: The nominal agent used by the hybrid agent.
    """

    def __init__(
            self,
            rl_agent: Type[RLBase],
            nominal_agent: Type[NominalBase]
    ):
        """
        Initializes the hybrid agent based on the provided reinforcement learning and nominal agents.
        
        Args:
            rl_agent: The reinforcement learning agent used by the hybrid agent.
            nominal_agent: The nominal agent used by the hybrid agent.
        """
        self.rl_agent = rl_agent
        self.nominal_agent = nominal_agent

    @abstractmethod
    def get_action(
        self, 
        obs: np.array, 
        deterministic: bool, 
        learning_started: bool
    ) -> Tuple[np.array, np.array]:
        """
        Computes the hybrid action.

        Gets both the nominal and rl actions from the agents and computes the mixed hybrid action. The RL agent outputs different actions
        depending on whether learning has started (random action if not) and whether the action should be deterministic (no action noise).

        Args:
            obs: The current observation.
            deterministic: Whether the rl agent's action should be deterministic.
            learning_started: Whether learning of the rl agent has started.
        
        Returns:
            A tuple of form (mixed_action, rl_action)
        """
        pass

    @abstractmethod
    def get_mixing_parameter(self) -> float:
        """
        Returns the current mixing parameter.

        This method should return the current mixing parameter of the hybrid agent.
        
        Returns:
            The current mixing parameter.
        """
        pass

    @abstractmethod
    def update_mixing_parameter(self, step: int):
        """
        Updates the mixing parameter.

        This method should update the mixing parameter of the hybrid agent. The update can be based on any logic, e.g. rl agent's uncertainty.
        It takes the argument step since we might randomize the beginning of the training. 
        """
        pass

    @abstractmethod
    def reset_mixing_parameter(self):
        """
        Used to reset the mixing parameter after episode.
        """
        pass

    @property
    @abstractmethod
    def get_uncertainty(self) -> float:
        """
        Used to receive the current uncertainty of the Agent.
        """
        raise NotImplementedError
