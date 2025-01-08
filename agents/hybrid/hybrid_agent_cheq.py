from agents.hybrid.hybrid_base import HybridBase
from agents.rl.rl_base import RLBase
from agents.rl.sac_ensemble import SACEnsemble
from agents.nominal.nominal_base import NominalBase
from typing import Type, Optional
import numpy as np
from collections import deque
import torch


class CHEQAgent(HybridBase):
    """
    Contextualized hybrid agent with changing mixing parameter.

    Hybrid agent that combines actions of the RL-agent and nominal-agent with a mixing parameter lambda.
    The merged action results from: a_mix = lambda * a_rl + (1 - lambda) * a_nominal.
    The lambda is dependent on the uncertainty estimate of the RL algorithm. 
    Thus, a value of lambda=1 corresponds to pure RL-action while a value of lambda=0 results in pure nominal action.

    We further add the option of uncertainty smoothing by allowing for an uncertainty-deque size.

    Attributes:
        rl_agent: The reinforcement learning agent used by the hybrid agent.
        nominal_agent: The nominal agent used by the hybrid agent.
    """
    
    def __init__(
            self,
            rl_agent: Type[SACEnsemble],
            nominal_agent: Type[NominalBase],
            uncertainty_min: float = 0.03,
            uncertainty_max: float = 0.15,
            uncertainty_deque: int = 1,
            weight_min: float = 0.2,
            weight_warmup_max: float = 0.3,
            weight_max: float = 1.0,
            warmup_steps: int = 5000, 
    ):
        """
        Initializes the hybrid agent based on the provided reinforcement learning and nominal agents.
        
        Args:
            rl_agent: The reinforcement learning agent used by the hybrid agent.
            nominal_agent: The nominal agent used by the hybrid agent.
        """
        self.rl_agent = rl_agent
        self.nominal_agent = nominal_agent

        super().__init__(rl_agent, nominal_agent)
        
        self.weight = None

        # saving for uncertainty computation
        self.uncertainty = None
        self.uncertainty_raw = None
        self.uncertainty_deque = deque(maxlen=uncertainty_deque)
        self.current_obs = None
        self.current_rl_action = None

        # define bounds and parameters for the clipped linear function
        self.uncertainty_min = uncertainty_min
        self.uncertainty_max = uncertainty_max
        self.weight_min = weight_min
        self.weight_warmup_max = weight_warmup_max
        self.weight_max = weight_max
        self.warmup_steps = warmup_steps
        
    def get_action(
            self,
            obs: torch.Tensor, 
            deterministic: bool = False, 
            learning_started: bool = True
    ):
        """
        Returns the hybrid and rl-action based on the provided observation.

        This method gets the rl-action from the rl-agent and the nominal-action from the nominal-agent and computes the mixed hybrid
        action according to a_mix = lambda * a_rl + (1 - lambda) * a_nominal.

        Args:
            obs: The current observation.
            deterministic: Whether the rl agent's action should be deterministic (no action noise).
            learning_started: Whether learning of the rl agent has started (otherwise action randomly sampled from action space).

        Note:
            This further saves the last observation and rl-action in the instance.

        Returns:
            A tuple of form (mixed_action, rl_action). This is necessary, since we want the rl_action for the buffer.
        """
        if isinstance(obs, np.ndarray):
            observation = torch.tensor(obs)
        elif isinstance(obs, torch.Tensor):
            observation = obs
        
        rl_action = self.rl_agent.get_action(obs=observation, deterministic=deterministic, learning_started=learning_started)
        nominal_action = self.nominal_agent.get_action(obs=observation)
        
        mixed_action = self.weight * rl_action + (1 - self.weight) * nominal_action

        # save current obs and current rl_action in instance of Agent 
        # so that it can be used for the update of the mixing parameter later on
        self.current_obs = observation
        self.current_rl_action = rl_action

        return mixed_action, rl_action
    
    def get_mixing_parameter(self) -> float:
        """
        Returns the mixing parameter lambda that was previously updated.
        """
        return self.weight

    
    def update_mixing_parameter(self, step: int) -> None:
        """
        Updates the mixing parameter lambda based on the uncertainty measure
        according to the clipped linear function.

        The uncertainty is computed using the q-ensemble.
        """
        # receive uncertainty from current rl agent and last state and action
        _, uncertainty = self.rl_agent.get_epistemic_uncertainty(state=self.current_obs, 
                                                                 action=self.current_rl_action)
        
        self.uncertainty_raw = uncertainty.detach().cpu().numpy()[0]

        # save current uncertainty in deque and compute uncertainty average
        self.uncertainty_deque.append(uncertainty.detach().cpu().numpy()[0])
        self.uncertainty = sum(self.uncertainty_deque)/len(self.uncertainty_deque)

        if step >= self.warmup_steps:
            # learning_starts already included in step (negative step)
            # update weight according to clipped function
            self.weight = self._clipped_linear_function()
        else:
            # random weight between weight_min and weight_warmup_max for the warm up until learning starts
            self.weight = np.random.uniform(low=self.weight_min, high=self.weight_warmup_max)

    def reset_mixing_parameter(self):
        """
        Resets the mixing parameter before every new episode.
        """
        self.weight = self.weight_min
        self.uncertainty_deque.clear()

    def _clipped_linear_function(self) -> float:
        """
        Private method:
        Computes the clipped linear function based on the uncertainty.
        Takes the bounds from the class instance.

        Returns:
            float: The new weight of the hybrid agent.
        """
        if self.uncertainty < self.uncertainty_min:
            new_weight = self.weight_max
        elif self.uncertainty > self.uncertainty_max:
            new_weight = self.weight_min
        else:
            new_weight = (self.uncertainty - self.uncertainty_max)/(self.uncertainty_min - self.uncertainty_max) * (self.weight_max - self.weight_min) + self.weight_min

        return new_weight
    
    ##### property-getter #####

    @property
    def get_uncertainty(self):
        """
        Returns the smoothed uncertainty of the Agent according to the deque.
        """
        return self.uncertainty
    
    @property
    def get_raw_uncertainty(self):
        """
        Returns the current uncertainty of the Agent.
        """
        return self.uncertainty_raw
    
