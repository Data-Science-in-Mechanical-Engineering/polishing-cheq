from agents.nominal.nominal_base import NominalBase
import numpy as np
from typing import Type, List
import gymnasium as gym
import torch


class NominalViaPoints(NominalBase):
    """
    Class of a nominal controller with constant impedance gains and movement along viapoints.

    This class implements a nominal controller that provides actions consisting of constant impedance gains and delta 
    position commands in the direction of the environment's next via point. Thereby, the nominal controller is designed to
    move the robot's end-effector along a predefined path of via points.

    Attributes:
        env: The environment in which the agent operates.
        indent: The indentation of the end-effector in the z-direction (into the surface).
        nominal_gains: The constant impedance gains for the nominal controller (stiffness, damping).
    """

    def __init__(
            self,
            env: Type[gym.Wrapper], 
            indent: float, 
            nominal_gains: List[float],
            device: torch.device = "cpu",
    ):
        """Initializes the nominal controller.
        
        Args:
            env: The environment in which the agent operates.
            indent: The indentation of the end-effector in the z-direction (into the surface).
            nominal_gains: The constant impedance gains for the nominal controller (stiffness, damping).
        """
        super().__init__(env)
        self.indent = indent
        self.nominal_gains = nominal_gains
        self.device = device

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Computes the nominal action based on the provided observation.
        
        This method computes the nominal action based on the current observation of the environment. The action consists of the 
        delta position to the next via point in x,y,z-direction and the constant impedance gains. The computed action is then 
        clipped to the action space of the environment.

        Args:
            obs: The current observation of the environment.

        Returns:
            The nominal action to be executed in the environment.
        """
        action = torch.zeros(self.env.unwrapped.action_space.shape)
        # NOT NICE - NEEDS TO CHANGE WHEN OBSERVATION CHANGES - SHOULD EXTRACT eef_pos FROM OBSERVATION
        eef_pos = obs[28:31]
        action[-12:] = torch.tensor(self.nominal_gains)
        action[:3] = torch.tensor(self.env.unwrapped.site_pos[:3]) - eef_pos[:3]
        action[3] -= torch.tensor(self.indent)

        # Clip action to environments action space
        action = torch.clip(action, torch.tensor(self.env.action_space.low), torch.tensor(self.env.action_space.high))

        return action.to(device=self.device) 