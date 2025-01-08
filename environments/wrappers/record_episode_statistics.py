# This code is taken and adapted from
# https://github.com/openai/gym/blob/master/gym/wrappers/record_episode_statistics.py

import gymnasium as gym
import numpy as np
from typing import Type, Union, Tuple


class RecordEpisodeStatistics(gym.Wrapper):
    """
    This class implements a wrapper that records observations across an episode 
    and outputs cumulative statistics at the end of each episode.
    
    At the end of the episode, this wrapper will add the aggregated statistics of the episode to the info dictionary
    using the key 'episode'. This statistics will include the following metrics:
        - return: the cumulative return of the episode
        - length: the number of steps taken in the episode
        - mean_force: the mean force applied during the episode
        - sd_force: the standard deviation of the force applied during the episode
        - max_force: the maximum force applied during the episode
        - min_force: the minimum force applied during the episode
        - mean_x_deviation: the mean deviation in the x axis during the episode
        - sd_x_deviation: the standard deviation of the deviation in the x axis during the episode
        - max_x_deviation: the maximum deviation in the x axis during the episode
        - min_x_deviation: the minimum deviation in the x axis during the episode
        - mean_y_velocity: the mean velocity in the y axis during the episode
        - sd_y_velocity: the standard deviation of the velocity in the y axis during the episode
        - max_y_velocity: the maximum velocity in the y axis during the episode
        - min_y_velocity: the minimum velocity in the y axis during the episode
        - num_wiped_markers: the number of markers wiped during the episode
    """

    def __init__(self, env: Type[gym.Env]):
        """
        Instantiate the RecordEpisodeStatistics wrapper.
        
        Args:
            env (Type[gym.Env]): The gym environment to wrap
        """
        super().__init__(env)
        self.episode_return = 0.0
        self.episode_length = 0
        self.episode_forces = []
        self.episode_x_deviations = []
        self.episode_y_velocities = []
        self.episode_wiped_markers = []

    def reset(self, **kwargs):
        """
        Reset the parent environment and the statistics.
        """
        observation = super().reset(**kwargs)

        self.episode_return = 0.0
        self.episode_length = 0
        self.episode_forces = []
        self.episode_x_deviations = []
        self.episode_y_velocities = []
        self.episode_wiped_markers = []

        return observation
    
    def step(self, action: Union[np.ndarray, float]) -> Tuple:
        """
        Execute an action in the environment and update the statistics.
        
        Args:
            action (np.ndarray, float): The action to execute in the environment.

        Returns:
            Tuple: observation, reward, terminated, truncated, info
        """

        obs, reward, terminated, truncated, info = self.env.step(action)  #, subpos, subforce, subjointvel, subtaus = self.env.step(action)

        assert isinstance(info, dict), \
            f"'info' dtype is {type(info)} while supported dtype is 'dict'. This may be due to usage of other wrappers in the wrong order."

        # append step to episode statistics
        self.episode_return += reward
        self.episode_length += 1
        self.episode_forces.append(info.get("force", 0.0))
        self.episode_x_deviations.append(info.get("x_dev", 0.0))
        self.episode_y_velocities.append(info.get("y_vel", 0.0))
        self.episode_wiped_markers.append(info.get("num_wiped_markers", 0))

        if terminated or truncated:
            # compute episode statistics
            episode_info = {
                "return": self.episode_return,
                "length": self.episode_length,
                "mean_force": np.mean(self.episode_forces),
                "sd_force": np.std(self.episode_forces),
                "max_force": np.max(self.episode_forces),
                "min_force": np.min(self.episode_forces),
                "mean_x_deviation": np.mean(self.episode_x_deviations),
                "sd_x_deviation": np.std(self.episode_x_deviations),
                "max_x_deviation": np.max(self.episode_x_deviations),
                "min_x_deviation": np.min(self.episode_x_deviations),
                "mean_y_velocity": np.mean(self.episode_y_velocities),
                "sd_y_velocity": np.std(self.episode_y_velocities),
                "max_y_velocity": np.max(self.episode_y_velocities),
                "min_y_velocity": np.min(self.episode_y_velocities),
                "num_wiped_markers": np.max(self.episode_wiped_markers),
            }


            info = {**info, **episode_info}

            # reset stats
            self.episode_return = 0.0
            self.episode_length = 0
            self.episode_forces = []
            self.episode_x_deviations = []
            self.episode_y_velocities = []
            self.episode_wiped_markers = []
        
        return obs, reward, terminated, truncated, info  #, subpos, subforce, subjointvel, subtaus
