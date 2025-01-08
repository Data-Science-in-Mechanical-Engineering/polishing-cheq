from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Dict


class RLBase(ABC):
    """
    Interface for the RL algorithm.
    All RL algorithms must implement the following methods to be compatible with upstream training loops.
    """

    @abstractmethod
    def get_action(self, obs: torch.Tensor, deterministic: bool = False, learning_started: bool = True) -> torch.Tensor:
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
        pass

    @abstractmethod
    def train(self, batch) -> Dict:
        """
        Trains the RL agent weights based on a batch of data.
        
        This method should update the RL agent's weights based on the provided batch of data taking one gradient step.
        The data should consist of a batch of transitions (s, a, r, s', done) sampled from the replay buffer.
        
        Note:
            This batch could also be enhanced with the Bernoulli-mask if ReplayBuffer is correclty initialized.

        Args:
            batch: A batch of data to train on.

        Returns:
            A dictionary containing the training metrics to be logged (e.g. q-loss, actor loss etc.).
        """
        pass

    @abstractmethod
    def update_params(self, params) -> None:
        """
        Updates the RL agent's network parameters.

        This method should update the RL agent's network parameters based on the provided parameter dictionary.
        More speecifically, this method is called within the asynchronous actor cycle to iteratively update the actor's
        weights with the weights of the learner's actor network given as a pytorch state_dict.

        Args:
            params: A dictionary containing the new network parameters to be updated.
        """
        pass

    @abstractmethod
    def save(self, save_path) -> None:
        """
        Saves the RL agent to the specified path.

        This method should save the RL agent's network parameters as well as the optimizer states to the specified path.

        Args:
            save_path: The path to save the RL agent to.
        """
        pass

    @abstractmethod
    def load(self, load_path) -> None:
        """
        Loads the RL agent from the specified path.

        This method should load the RL agent's network parameters as well as the optimizer states from the specified path.

        Args:
            load_path: The path to load the RL agent from.
        """
        pass
