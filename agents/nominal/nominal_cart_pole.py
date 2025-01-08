from agents.nominal.nominal_base import NominalBase

from typing import Type
from torch import nn
from torch.nn import init

import gymnasium as gym
import numpy as np
import torch
import control as ct

class NominalCartPole(NominalBase):
    """
    Class of a nominal controller for the cart_pole environment.

    Attributes:
        env: The environment in which the agent operates.
    """
    def __init__(
            self,
            env: Type[gym.Env],
            device: torch.device = 'cpu',
            action_skew: float = 0.0,
            silly: bool = False,
            place_poles: bool = False,
            K: any = None
    ):
        """
        Initialize the nominal cart_pole controller.

        Args:
            env (Type[gym.Env]): The environment in which the agent operates.
        """
        super().__init__(env)

        self.device = device

        cartpole = ct.NonlinearIOSystem(
            updfcn=self.env.unwrapped.ct_sys_update, 
            outfcn=self.env.unwrapped.ct_sys_output, 
            states=4, 
            name="cartpole",
            inputs=["action"],
            outputs=["x", "x_dot", "theta", "theta_dot"]
        )
        linsys = cartpole.linearize(x0=self.env.unwrapped.goal_state, u0=np.array([0.]))
        linsys_d = linsys.sample(self.env.unwrapped.tau)

        cost_x = self.env.unwrapped.cost_x
        cost_x_dot = self.env.unwrapped.cost_x_dot
        cost_theta = env.unwrapped.cost_theta
        cost_theta_dot = env.unwrapped.cost_theta_dot
        cost_control = env.unwrapped.cost_control

        Q = np.diag([cost_x, cost_x_dot, cost_theta, cost_theta_dot])
        R = np.diag([cost_control])

        if place_poles:
            self.K = self.place_poles(linsys_d)
        elif K:
            self.K = torch.tensor(K)
        else:
            self.K = torch.tensor(ct.lqr(linsys_d, Q, R)[0])

        self.fc = nn.Linear(in_features=4, out_features=1)
        weights = self.K

        with torch.no_grad():
            self.fc.weight.copy_(-weights)
            init.constant_(self.fc.bias, 0)

        # possible deviations for controller impurities
        self.action_skew = action_skew if action_skew else 0.0
        self.silly = silly

        self.max_action = nn.Parameter(torch.tensor(env.action_space.high))
        self.min_action = nn.Parameter(torch.tensor(env.action_space.low))

    def place_poles(self, linsys):
        desired_poles = [0.35114616 + 0.1j, 0.35114616 - 0.1j, 0.97800293 + 0.01737253j, 0.97800293 - 0.01737253j]
        K = ct.place(linsys.A, linsys.B, desired_poles)
        return torch.tensor(K[0])

    def forward(self, x: torch.Tensor):
        """
        This method is called when the Controller __call__ is applied.
        In order to include impurities of action, it holds the option of 
        adding an action_skew term that alters the outcome of the controller.
        Further, if silly=True, it only takes the action_skew as output.
        The action_skew and silly are initialized in the __init__.

        Args:
            x (torch.Tensor): The current state

        Returns:
            action: The outcome of the controller action.
        """
        if self.silly:
            output = torch.tensor(self.action_skew)
            output = torch.clip(output, self.min_action, self.max_action)
        else:
            output = self.fc(x)
            # apply possible skew before the clipping
            output = output + torch.tensor(self.action_skew)
            output = torch.clip(output, self.min_action, self.max_action)

        return output.to(self.device)

    def get_action(self, obs: torch.Tensor, included_weight: bool = True):
        """
        Gets the action using a forward run through the controller.
        
        Note:
            This method is also able to check if a weight is included in the vector.
            If there is a weight included, will delete it from the state-vector.

        Args:
            obs (torch.Tensor): The current state of the system.
            included_weight (bool, optional): If the state-vector includes a weight. Defaults to True. Not in use currently...

        Returns:
            torch.Tensor: The action corresponding to the weight.
        """
        if len(obs) == 5:
            # delete weight from a copy of the state vector because we dont need it for the controller.
            state = obs[:-1]
        else:
            state = obs

        return self.forward(x=state)