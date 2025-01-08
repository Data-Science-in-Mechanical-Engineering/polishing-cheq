import torch
import numpy as np
import gymnasium as gym
from typing import Type
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from agents.nominal.nominal_base import NominalBase


class NominalRobotDummy(NominalBase):
    """
    This controller is only meant for experimental purposes to find out some bounds.

    Args:
        NominalBase (class): The general parent of controls
    """

    def __init__(     
            self, 
            env: Type[gym.Wrapper],
            nominal_gains: list[float],
            action: list[float],
            indent: float,
            device: torch.device = "cpu",
    ):
        """
        Initialization of the nominal agent based on the provided environment.
        """
        super().__init__(env)
        self.env = env
        self.nominal_gains = nominal_gains
        self.action = action
        self.indent = indent
        self.device = device
        

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Computes the next action

        Args:
            obs (torch.Tensor): The current observation / state of the robot

        Returns:
            torch.Tensor: The next action to perform.
        """
        action = torch.zeros(self.env.unwrapped.action_space.shape)

        # computation of eef_pos with indent ==================================
        current_rotation = R.from_euler(seq="xyz", angles=R.from_quat(obs[31:35]).as_euler("xyz", degrees=True), degrees=True)
        z_new = current_rotation.apply(np.array([0, 0, 1]))
        indent_rot = self.indent * z_new
        self.indent_rot = indent_rot
        eef_pos = obs[28:31] + self.indent_rot
        # ================================== computation of eef_pos with indent

        # action computation for pos and gains ================================
        action[-7:] = torch.tensor(self.nominal_gains)
        next_pos = eef_pos + self.action
        action[:3] = torch.tensor(next_pos - eef_pos)
        print(f"controller sees eef_pos: {eef_pos}")
        # ================================ action computation for pos and gains

        # compuation of orientational shift ===================================
        if self.env.unwrapped.control_mode == "pose":
            curr_quat = obs[31:35]  # current quaternion
            trans = R.from_euler(seq="xyz", angles=R.from_quat(curr_quat).as_euler("xyz", degrees=True) - [0,0,-0.1], degrees=True) * R.from_quat(curr_quat).inv()  # computation of necessary transformation
            action[3:6] = torch.tensor(trans.as_euler("xyz", degrees=True))  # give transformation as euler angle
        else:
            pass
        # =================================== compuation of orientational shift
        
        action = torch.clip(action, torch.tensor(self.env.unwrapped.action_space.low), torch.tensor(self.env.unwrapped.action_space.high))
        
        return action.to(device=self.device)
    
    @property
    def get_curr_indent(self):
        return self.indent_rot
    