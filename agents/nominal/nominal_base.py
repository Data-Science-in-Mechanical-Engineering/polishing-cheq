from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Type
import gymnasium as gym



class NominalBase(ABC):
    """
    Abstract base class defining an interface for nominal agents.

    Abstract base class defining an interface for nominal agents that compute a control action based on the environment's 
    state following predefined rules. All nominal agents must implement this class to be compatible with upstream training loops. 
    
    Attributes:
        env: The environment in which the agent operates.
    """

    def __init__(self, env: Type[gym.Env]):
        """Initializes the nominal agent based on the provided environment.

        Args:
            env: The environment in which the agent operates.
        """
        self.env = env

    @abstractmethod
    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Computes the nominal action based on the provided observation.

        This method should implement the main logic of the nominal controller computing a nominal control action based on 
        the current observation of the environment.

        Args:
            obs: The current observation of the environment.

        Returns:
            The nominal action to be executed in the environment.
        """
        pass
