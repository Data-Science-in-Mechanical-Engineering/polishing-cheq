import numpy as np
import torch
import gymnasium as gym
import time
from typing import Union, Optional, Literal
from omegaconf import ListConfig

from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from scipy.spatial.transform import Rotation as R
import scipy.stats as stats
from itertools import cycle
from utils.launcher_utils import convert_to_dict
from utils.reward_utils import _compute_penalty, _compute_gaussian_reward, _compute_linear_reward, _normalize_weighting, _compute_piecewise_linear_quad_reward
from environments.controllers import opspace
from environments.mujoco_gym_env import MujocoGymEnv
from pathlib import Path
import mujoco
# For interactive mode:
import mujoco.viewer as mjv

_HERE = Path(__file__).parent
interactive_mode = False


#TODO on/off rendering
class SimRobotEnv(MujocoGymEnv):

    def __init__(self, env_config, task_config):
        """
        Initialize the SimRobotEnv which inherits from MujocoGymEnv.

        Args:
            env_config (dict): The config-dict of the environment
            task_config (dict): The config-dict of the task
        """
        self.env_config = env_config
        arena_path = _HERE / "models" / f"{self.env_config.arena}.xml"

        super().__init__(xml_path=arena_path)

        self.task_config = task_config
        self.reward_config = self.task_config.reward_func
        self.reward_type = self.reward_config.name
        self.task_type = self.task_config.type

        # Initialization of variables ----------------
        self.environment_step = 0
        self.reward_done = 0.0
        self.currstate = dict(joint_pos=np.zeros(7),
                              joint_vel=np.zeros(7),
                              joint_torque=np.zeros(7),
                              eef_pos=np.zeros(3),
                              eef_quat=np.zeros(4),
                              eef_vel_trans=np.zeros(3),
                              eef_vel_rot=np.zeros(3),
                              eef_force=np.zeros(3),
                              eef_torque=np.zeros(3))
        self.ee_force_bias = np.zeros(6)
        self.total_force_ee = 0
        self.force_buffer = []
        self.episodic_return = 0
        self.min_y_pos = np.inf
        self.reward_info = {}
        self.wiped_markers = []
        self.episode_forces = []
        self.f_excess = 0

        #TODO: Collision Detection ?

        # Caching ----------------------------------------------------
        self._panda_dof_ids = np.asarray(
            [self._model.joint(f"joint{i}").id for i in range(1, 8)]
        )
        self._panda_ctrl_ids = np.asarray(
            [self._model.actuator(f"actuator{i}").id for i in range(1, 8)]
        )
        # self._gripper_ctrl_id = self._model.actuator("fingers_actuator").id
        self._pinch_site_id = self._model.site("grip_site").id
        # --------------------------------------------------- Caching

        # Control related Config ----------------
        self.horizon = self.env_config.horizon
        self.reset_poses = dict(init_pose=convert_to_dict(self.env_config.reset_poses.init_pose))
        self.init_pose = np.array([val for val in self.reset_poses["init_pose"].values()])
        self.mixed_action_env = self.env_config.mixed_action_env
        # ---------------- Control related Config

        # Safety related Config ----------------
        self.pos_cap = self.task_config.pos_cap
        self.vel_cap = self.task_config.vel_cap
        self.force_cap = self.task_config.force_cap
        # ---------------- Safety related Config

        # Site Config --------------------------
        self.dist_th = self.task_config.dist_th
        self.dist_wiped = self.task_config.dist_wiped
        self.sites = self.task_config.sites
        self.curr_site = list(self.sites)[0]
        self.goal_pos = self.sites[list(self.sites)[-1]].xpos  # last site
        self.site_cycle = cycle(list(self.sites))
        self.site_pos = self.sites[self.curr_site].xpos
        self.site_dir = self.sites[self.curr_site].dir
        for site_name, site in dict(self.sites.copy()).items():
            self._model.body(site_name).pos = site.xpos
        # -------------------------- Site Config

        # Orientation control related Config ----------------
        self.control_mode = self.task_config.control_mode
        assert (self.control_mode in ["position",
                                      "pose"]), f"Control mode {self.control_mode} is not supported. Must be one of ['position', 'pose']."

        self.fixed_orientation = self.task_config.fixed_orientation
        if self.control_mode == "position":
            assert (self.fixed_orientation is not None), "Fixed orientation must be specified for position control."
            assert (len(self.fixed_orientation) == 4), f"Fixed orientation must be a valid quaternion of length 4. Got {len(self.fixed_orientation)}."
        # ---------------- Orientation control related Config

        # REWARD RELATED CONFIG -----------------------------------------------
        # basics
        self.target_force = self.task_config.target_force
        self.target_wvel = self.task_config.target_vel
        self.min_force_for_contact = self.task_config.min_force_for_contact
        assert (self.min_force_for_contact < self.target_force)
        self.arm_limit_collision_penalty = self.task_config.arm_limit_collision_penalty
        self.reward_calc_c_done = self.task_config.reward_calc_c_done
        self.penalty_truncating = self.task_config.penalty_truncating

        if self.reward_type == "penalty":
            # initialization for penalty reward (old version)
            self.reward_calc_upper_limit_force = self.reward_config.reward_calc_upper_limit_force
            self.reward_calc_lower_limit_force = self.reward_config.reward_calc_lower_limit_force
            assert (self.target_force < self.reward_calc_upper_limit_force)
            assert (self.target_force > self.reward_calc_lower_limit_force)
            self.target_xvel = self.reward_config.target_xvel
            self.reward_calc_upper_limit_xvel = self.reward_config.reward_calc_upper_limit_xvel
            self.reward_calc_lower_limit_xvel = self.reward_config.reward_calc_lower_limit_xvel
            assert (self.target_xvel < self.reward_calc_upper_limit_xvel)
            assert (self.target_xvel > self.reward_calc_lower_limit_xvel)
            self.target_xdist = self.reward_config.target_xdist
            self.reward_calc_upper_limit_xdist = self.reward_config.reward_calc_upper_limit_xdist
            self.reward_calc_lower_limit_xdist = self.reward_config.reward_calc_lower_limit_xdist
            assert (self.reward_calc_lower_limit_xdist < self.reward_calc_upper_limit_xdist)
            self.reward_calc_upper_limit_wvel = self.reward_config.reward_calc_upper_limit_vel
            self.reward_calc_lower_limit_wvel = self.reward_config.reward_calc_lower_limit_vel
            assert (self.target_wvel < self.reward_calc_upper_limit_wvel)
            assert (self.target_wvel > self.reward_calc_lower_limit_wvel)
            self.target_dist_wipe = self.reward_config.target_dist_wipe
            self.reward_calc_upper_limit_dist_wipe = self.reward_config.reward_calc_upper_limit_dist_wipe
            self.reward_calc_lower_limit_dist_wipe = self.reward_config.reward_calc_lower_limit_dist_wipe
            # weightings
            self.reward_calc_c_force = self.reward_config.reward_calc_c_force
            self.reward_calc_c_xvel = self.reward_config.reward_calc_c_xvel
            self.reward_calc_c_xdist = self.reward_config.reward_calc_c_xdist
            self.reward_calc_c_wvel = self.reward_config.reward_calc_c_vel
            self.reward_calc_c_dist_wipe = self.reward_config.reward_calc_c_dist_wipe
            self.reward_calc_c_contact = self.reward_config.reward_calc_c_contact

        elif self.reward_type == "gaussian":
            # initialization for gaussian reward
            self.theta_force = np.array(self.reward_config.theta_force)
            self.theta_cross = np.array(self.reward_config.theta_cross)
            self.theta_dir = np.array(self.reward_config.theta_dir)
            self.theta_vel = np.array(self.reward_config.theta_vel)
            for key, val in self.__dict__.items():
                if key.startswith("theta"):
                    assert isinstance(val, np.ndarray) and len(val)==2
                    assert all(val > 0.0)
                    
        elif self.reward_type == "linear":
            # initialization for linear reward
            self.reward_calc_upper_limit_force = self.reward_config.reward_calc_upper_limit_force
            self.reward_calc_lower_limit_force = self.reward_config.reward_calc_lower_limit_force
            assert (self.target_force < self.reward_calc_upper_limit_force)
            assert (self.target_force > self.reward_calc_lower_limit_force)
            self.target_cross_error = self.reward_config.target_cross_error
            self.reward_calc_upper_limit_cross = self.reward_config.reward_calc_upper_limit_cross
            self.reward_calc_lower_limit_cross = self.reward_config.reward_calc_lower_limit_cross
            assert self.reward_calc_upper_limit_cross > 0.0
            assert self.reward_calc_lower_limit_cross < 0.0
            self.target_dir_error = self.reward_config.target_dir_error
            self.reward_calc_upper_limit_dir = self.reward_config.reward_calc_upper_limit_dir
            self.reward_calc_lower_limit_dir = self.reward_config.reward_calc_lower_limit_dir
            assert self.reward_calc_upper_limit_dir > 0.0
            assert self.reward_calc_lower_limit_dir < 0.0
            self.target_vel_error = self.reward_config.target_vel_error
            self.reward_calc_upper_limit_vel_error = self.reward_config.reward_calc_upper_limit_vel_error
            self.reward_calc_lower_limit_vel_error = self.reward_config.reward_calc_lower_limit_vel_error
            assert self.reward_calc_upper_limit_vel_error > 0.0
            assert self.reward_calc_lower_limit_vel_error < 0.0
            # weightings
            self.reward_calc_c_force = self.reward_config.reward_calc_c_force
            self.reward_calc_c_cross = self.reward_config.reward_calc_c_cross
            self.reward_calc_c_dir = self.reward_config.reward_calc_c_dir
            self.reward_calc_c_vel_error = self.reward_config.reward_calc_c_vel_error
        elif self.reward_type == "real_robot_reward":
            # initialization for real robot reward
            # based on error value of force --------->
            self.reward_calc_upper_limit_force_inner = self.reward_config.reward_calc_upper_limit_force_inner
            self.reward_calc_lower_limit_force_inner = self.reward_config.reward_calc_lower_limit_force_inner
            self.reward_calc_upper_limit_force_outer = self.reward_config.reward_calc_upper_limit_force_outer
            self.reward_calc_lower_limit_force_outer = self.reward_config.reward_calc_lower_limit_force_outer
            # <--------- based on error value of force
            self.target_cross_error = self.reward_config.target_cross_error
            self.reward_calc_upper_limit_cross = self.reward_config.reward_calc_upper_limit_cross
            self.reward_calc_lower_limit_cross = self.reward_config.reward_calc_lower_limit_cross
            assert self.reward_calc_upper_limit_cross > 0.0
            assert self.reward_calc_lower_limit_cross < 0.0
            self.target_dir_error = self.reward_config.target_dir_error
            self.reward_calc_upper_limit_dir = self.reward_config.reward_calc_upper_limit_dir
            self.reward_calc_lower_limit_dir = self.reward_config.reward_calc_lower_limit_dir
            assert self.reward_calc_upper_limit_dir > 0.0
            assert self.reward_calc_lower_limit_dir < 0.0
            self.target_vel_error = self.reward_config.target_vel_error
            self.reward_calc_upper_limit_vel_error = self.reward_config.reward_calc_upper_limit_vel_error
            self.reward_calc_lower_limit_vel_error = self.reward_config.reward_calc_lower_limit_vel_error
            assert self.reward_calc_upper_limit_vel_error > 0.0
            assert self.reward_calc_lower_limit_vel_error < 0.0
            # weightings
            self.reward_calc_c_force = self.reward_config.reward_calc_c_force
            self.reward_calc_c_cross = self.reward_config.reward_calc_c_cross
            self.reward_calc_c_dir = self.reward_config.reward_calc_c_dir
            self.reward_calc_c_vel_error = self.reward_config.reward_calc_c_vel_error
        else:
            raise ValueError("Please name a reward_function that is defined for the __init__ call.")
        
        # ----------------------------------------------- REWARD RELATED CONFIG

        # Initialize action and observation space -------------
        self.last_point = np.array(self.task_config.last_point_init)
        self.corresponding_path_points = np.array(self.task_config.corresponding_path_points_init)
        self.corresponding_path_dirs = np.array(self.task_config.corresponding_path_dirs_init)
        self.corresponding_path_ori = np.array(self.task_config.corresponding_path_ori_init)
        self.state_extension = self.task_config.state_space_ext
        self.action_space = self._initialize_action_space()
        self.observation_space = self._initialize_observation_space()
        # ------------- Initialize action and observation space

        # Initialize simulation and renderer -----------------------
        cam_config = convert_to_dict(self.env_config.cam_config)

        self._model.vis.global_.offwidth = self.env_config.image_size_width
        self._model.vis.global_.offheight = self.env_config.image_size_height
        self.render_mode = self.env_config.render_mode
        self.renderer = MujocoRenderer(self._model, self._data, default_cam_config=cam_config)
        self.viewer = self.renderer._get_viewer(self.render_mode)

        if interactive_mode:
            self.sim = mjv.launch_passive(self._model, self._data, show_left_ui=False, show_right_ui=False)
            self.sim.cam.azimuth = cam_config["azimuth"]
            self.sim.cam.elevation = cam_config["elevation"]
            self.sim.cam.lookat = cam_config["lookat"]
        # ------------------------ Initialize simulation and renderer

    def _initialize_action_space(self) -> gym.spaces.Box:
        """
        Initialize the action space using configuration of the task_config.

        Returns:
            gym.spaces.Box: The box of the action space.
        """
        # scaling parameters
        scale_lower_bound = self.task_config.scale_lower_bound
        scale_higher_bound = self.task_config.scale_higher_bound

        # Get action space for delta position actions
        position_actions_min = self.task_config.position_actions_min
        position_actions_max = self.task_config.position_actions_max

        # Get action space for stiffness and damping gains
        stiffness_actions_min = self.task_config.kp_min
        stiffness_actions_max = self.task_config.kp_max

        damping_actions_min = [self.task_config.damping_factor_min]
        damping_actions_max = [self.task_config.damping_factor_max]

        # distinguish between position or pose setup
        if self.task_config.control_mode == "position":
            low = np.concatenate((position_actions_min, stiffness_actions_min, damping_actions_min))
            high = np.concatenate((position_actions_max, stiffness_actions_max, damping_actions_max))
        elif self.task_config.control_mode == "pose":
            # Get action space for delta orientation actions
            orientation_actions_min = self.task_config.orientation_actions_min
            orientation_actions_max = self.task_config.orientation_actions_max

            low = np.concatenate(
                (position_actions_min, orientation_actions_min, stiffness_actions_min, damping_actions_min))
            high = np.concatenate(
                (position_actions_max, orientation_actions_max, stiffness_actions_max, damping_actions_max))
        else:
            raise ValueError(
                f"Control mode {self.task_config.control_mode} is not supported. Must be one of ['position', 'pose'].")
        
        # scale action space according to min-max scaler and remember transformation for later
        def scaling_action(action: np.ndarray) -> np.ndarray:
            """Scales action based on the transformation from the environment initialization."""
            return (scale_higher_bound - scale_lower_bound) * (action - low)/(high - low) + scale_lower_bound
        def rescaling_action(scaled_action: np.ndarray) -> np.ndarray:
            """Rescales action based on the transformation from the environment initialization."""
            return (high - low) * (scaled_action - scale_lower_bound)/(scale_higher_bound - scale_lower_bound) + low
        self.scaling_func = scaling_action
        self.rescaling_func = rescaling_action
        
        scaled_low = scaling_action(low)
        scaled_high = scaling_action(high)

        action_space = gym.spaces.Box(low=scaled_low, high=scaled_high, dtype=np.float32)
        return action_space

    def _initialize_observation_space(self) -> gym.spaces.Box:
        """
        Initialize the observation space by looking at the get_observation initialization.

        Returns:
            gym.spaces.Box: The box of the observation space.
        """
        obs = self._get_observation()
        low = -np.inf * np.ones(obs.shape)
        high = np.inf * np.ones(obs.shape)
        if self.mixed_action_env:
            low = np.append(low, 0.0)
            high = np.append(high, 1.0)

        observation_space = gym.spaces.Box(low=low, high=high, shape=high.shape, dtype=np.float32)
        return observation_space

    def _update_currstate(self):

        self.currstate["joint_pos"] = np.array([self._data.sensor(f"joint{i}_pos").data for i in range(1,8)]).ravel()
        self.currstate["joint_vel"] = np.array([self._data.sensor(f"joint{i}_vel").data for i in range(1,8)]).ravel()
        self.currstate["eef_pos"] = self._data.sensor("eef_pos").data
        self.currstate["eef_quat"] = self._data.sensor("eef_quat").data
        self.currstate["eef_vel_trans"] = self._data.sensor("eef_vel").data.T
        self.currstate["eef_vel_rot"] = self._data.sensor("eef_vel_rot").data
        self.currstate["eef_force"] = self._median_force_computation()
        self.currstate["eef_torque"] = self._data.sensor("eef_torque").data

    def _get_observation(self) -> np.ndarray:
        """
        Gets the current observation by checking the current sensor data.
        Concatenates some parts of the sensor data to a state and outputs this 
        array as the new state.

        Returns:
            np.ndarray: The updated current state.
        """
        # Update current state
        self._update_currstate()
        # Extract relevant information
        obs = np.concatenate((
            self.currstate["joint_pos"],
            np.cos(self.currstate["joint_pos"]),
            np.sin(self.currstate["joint_pos"]),
            self.currstate["joint_vel"],
            self.currstate["eef_pos"],
            self.currstate["eef_quat"],
            self.currstate["eef_vel_trans"],
            self.currstate["eef_vel_rot"],
            self.currstate["eef_force"] - self.ee_force_bias[:3],
        ))
        if self.state_extension == "traj":
            # append obs with trajectory point differences
            obs_traj = []
            for pos, dir in zip(self.corresponding_path_points, self.corresponding_path_dirs):
                curr_dir = np.array(self.currstate["eef_pos"]) - self.last_point
                obs_traj.append(pos - np.array(self.currstate["eef_pos"]))
                obs_traj.append(self.target_wvel * dir - curr_dir)
            obs = np.concatenate((obs, np.array(obs_traj).flatten()))
        elif self.state_extension == "via":
            # append obs with difference to next via point specifics
            curr_dir = np.array(self.currstate["eef_pos"]) - self.last_point
            obs_via = np.concatenate((
                np.array(self.site_pos) - np.array(self.currstate["eef_pos"]),  # distance to next via point
                self.target_wvel * np.array(self.site_dir) - np.array(curr_dir),  # directed velocity difference to next via point
            ))
            obs = np.concatenate((obs, obs_via))
        else:
            pass

        return obs

    def _get_info(self) -> dict[str, int]:
        """
        The private method to create the info for the step output.
        Takes information of the current state and other class attributes to create a dict

        Returns:
            dict[str, int]: The information dictionary.
        """
        info = dict()
        info["force"] = self.total_force_ee
        info["x_pos"] = self.currstate["eef_pos"][0]
        info["y_pos"] = self.currstate["eef_pos"][1]
        info["z_pos"] = self.currstate["eef_pos"][2]
        info["x_vel"] = self.currstate["eef_vel_trans"][0]
        info["y_vel"] = self.currstate["eef_vel_trans"][1]
        info["z_vel"] = self.currstate["eef_vel_trans"][2]
        info["x_dev"] = np.abs(self.currstate["eef_pos"][0] - self.goal_pos[0])
        info["min_y_pos"] = self.min_y_pos
        info["return"] = self.episodic_return
        info["episode_forces"] = self.episode_forces  # this is only the absolute value
        info["num_wiped_markers"] = len(self.wiped_markers)
        info["episodic_length"] = self.environment_step
        info["reward_info"] = {key: val/self.environment_step for (key,val) in self.reward_info.items()}

        return info

    def reward(self, truncated: bool) -> float:
        """
        The reward function that distinguishes between different reward_types.
        Further, adds the reward for done or the penalty for truncated.
        """
        # distinguish between different reward functions
        if self.reward_type == "gaussian":
            reward, new_reward_info = self._reward_gaussian()
        elif self.reward_type == "penalty":
            reward, new_reward_info = self._reward_penalty()
        elif self.reward_type == "linear":
            reward, new_reward_info = self._reward_linear()
        elif self.reward_type == "real_robot_reward":
            reward, new_reward_info = self._reward_real_robot()
        else:
            raise ValueError("Please assure that you enter the correct reward_type.")

        # Compute task done reward
        self.reward_done = 0.0
        update = self._update_wiped_markers()

        if update:
            if self.wiped_markers[-1] == list(self.sites)[-1]:  
                # check if we wiped the last marker
                self.reward_done = self.task_config.reward_done
        reward += self.reward_done * self.reward_calc_c_done

        # Compute penalty for truncated
        reward -= self.penalty_truncating if truncated else 0.0

        # for logging
        self.reward_info.update(new_reward_info)

        return reward

    def step(self, action: Union[np.ndarray, torch.Tensor]) -> tuple:
        """
        The environment step given an action.

        Args:
            action (np.ndarray, torch.Tensor): The action to take

        Returns:
            tuple: obs, reward, terminated, truncated, info
        """
        # clip action to action space
        if isinstance(action, np.ndarray):
            action = np.clip(action, self.action_space.low, self.action_space.high)
        elif isinstance(action, torch.Tensor):
            if action.get_device() != -1:
                action = action.cpu()
            else:
                pass
            action = torch.clip(action, torch.tensor(self.action_space.low), torch.tensor(self.action_space.high)).detach().cpu().numpy()
        else:
            raise Exception(f"Expected action to be numpy or torch but is {type(action)}")
        
        rescaled_action = self.rescaling_func(scaled_action=action)

        # transform delta pose actions to absolute pose actions
        next_pose = np.concatenate((self.currstate["eef_pos"], self.currstate["eef_quat"]))
        xyz_delta = rescaled_action[:3] # In Serl, this is scaled
        next_pose[:3] = next_pose[:3] + xyz_delta #np.clip(next_pose[:3] + xyz_delta, *_CARTESIAN_BOUNDS)
        self._data.mocap_pos[0] = next_pose[:3]

        # orientation computation
        if self.control_mode == "position":
            next_pose[3:] = self.fixed_orientation
        elif self.control_mode == "pose":
            pose_action = R.from_euler("xyz", rescaled_action[3:6], degrees=True)
            curr_quat = R.from_quat(self.currstate["eef_quat"], scalar_first=True)
            next_pose[3:] = (pose_action * curr_quat).as_quat(scalar_first=True)
        self._data.mocap_quat[0] = next_pose[3:]

        self.force_buffer = []
        subpos = []
        subjointvels = []
        taus = []
        # Wait for the environment to evolve
        for _ in range(self._n_substeps):
            # using aperiodic damping coefficients based on stiffness
            tau = opspace(
                model=self._model,
                data=self._data,
                site_id=self._pinch_site_id,
                dof_ids=self._panda_dof_ids,
                pos=self._data.mocap_pos[0],
                ori=self._data.mocap_quat[0],
                stiffness=rescaled_action[6:-1],
                damping_ratio=rescaled_action[-1],
                joint=self.init_pose,
                gravity_comp=True,
            )
            self._data.ctrl[self._panda_ctrl_ids] = tau
            mujoco.mj_step(self._model, self._data)

            self.force_buffer.append(np.array(self._data.sensor("eef_force").data - self.ee_force_bias[:3]))
            subpos.append(np.array(self._data.sensor("eef_pos").data))
            subjointvels.append(np.array([self._data.sensor(f"joint{i}_vel").data for i in range(1,8)]).ravel())
            taus.append(np.array(tau))
        
        if interactive_mode:
            self.sim.sync()

        # Get observation, check truncated or terminated and compute reward
        obs = self._get_observation()
        truncated = self._truncated()
        reward = self.reward(truncated=truncated)

        # Update the next site
        self._update_next_site()

        terminated = self._terminated()

        # Increment step counter
        if self.currstate["eef_pos"][1] < self.min_y_pos:
            # for plotting minimal y_pos value of episode
            self.min_y_pos = self.currstate["eef_pos"][1]
        self.environment_step += 1
        self.episodic_return += reward
        self.last_point = np.array(self.currstate["eef_pos"])

        # Fill info dict for logging
        info = self._get_info()
        info.update({"return": self.episodic_return})

        return obs, reward, terminated, truncated, info  #, subpos, self.force_buffer, subjointvels, taus

    def reset(self, **kwargs) -> tuple:
        """
        Resets the environment.

        Returns:
            tuple: obs, {}
        """
        super(SimRobotEnv, self).reset(**kwargs)
        mujoco.mj_resetData(self._model, self._data)

        # Set force bias for next episode
        self._update_currstate()
        self.ee_force_bias = np.concatenate((self.currstate["eef_force"],
                                             self.currstate["eef_torque"]))
        self.ee_force_bias = np.zeros(6)

        #Reset arm to home pos
        self._data.qpos[self._panda_dof_ids] = self.init_pose
        mujoco.mj_forward(self._model, self._data)

        # Reset mocap body to home position.
        tcp_pos = self._data.sensor("eef_pos").data
        self._data.mocap_pos[0] = tcp_pos

        # Sample a new block position.
        #block_xy = np.random.uniform(*_SAMPLING_BOUNDS)
        #self._data.jnt("block").qpos[:3] = (*block_xy, self._block_z)
        mujoco.mj_forward(self._model, self._data)

        # Reset environment variables
        self.environment_step = 0
        self.episodic_return = 0
        self.wiped_markers = []
        self.min_y_pos = np.inf
        self.episode_forces = []
        self.force_buffer = []
        self.collisions = 0
        self.f_excess = 0
        self.curr_site = list(self.sites)[0]
        self.site_cycle = cycle(list(self.sites))
        self.site_pos = self.sites[self.curr_site].xpos
        self.site_dir = self.sites[self.curr_site].dir

        self.reward_info = {}

        # Get observation
        obs = self._get_observation()

        return obs, {}

    def render(self):
        return self.viewer.render(render_mode=self.render_mode).astype(np.short)

    def close(self):
        raise NotImplementedError()

    def _terminated(self) -> bool:
        """
        Computation of if the task is done (terminated).
        This includes the check if all markers were wiped or if the horizon is reached.

        Returns:
            bool
        """
        terminated = False

        if self.reward_done == self.task_config.reward_done:
            terminated = True
        elif self.environment_step >= self.horizon:
            terminated = True
        else:
            pass

        return terminated

    def _truncated(self) -> bool:
        """
        Computes if the episode is truncated or not.
        This is done by checking limits of the config
        (horizon, joint_limits, pos_limits, etc.).

        Returns:
            bool
        """
        truncated = False

        #TODO: proper truncating, What is meant by that?
        if self._check_joint_limits():
            print(f'Joint limits exceeded. Truncating episode at step {self.environment_step}.')
            truncated = True
        if self._check_collision():
            print(f'Collision detected. Truncating episode at step {self.environment_step}.')
            truncated = True
        if self._check_cartesian_pos_limits():
            print(f'Cartesian position limits exceeded. Truncating episode at step {self.environment_step}.')
            truncated = True
        if self._check_cartesian_vel_limits():
            print(f'Cartesian velocity limits exceeded. Truncating episode at step {self.environment_step}.')
            truncated = True
        if self._check_orientation_limits():
            print(f'Orientation limits exceeded. Truncating episode at step {self.environment_step}.')
            truncated = True
        if self._check_eef_force_limits():
            print(f'End effector force limits exceeded. Truncating episode at step {self.environment_step}.')
            truncated = True

        return truncated

    ##### ENVIRONMENT SPECIFIC HELPER FUNCTIONS #####
    
    def _update_wiped_markers(self) -> bool:
        """
        Update of the wiped markers.
        It updates the wiped markers if there is a small distance to the current target
        and there is a force bigger than the specified contact force.
        Only checks for the sites excepts the last one.

        Performs different checks for different tasks.

        IMPROVEMENTS: CAN BE OPTIMIZED BY AVOIDING LOOPS AND USING VECTORIZED COMPUTATIONS

        Returns:
            bool: When there is an update returns True, else returns False.
        """
        active_markers = []

        if self.task_type == "2D":
            # Only go into this computation if robot in contact with the table
            # cause then we can compare z-coord
            if self.total_force_ee >= self.min_force_for_contact and np.linalg.norm(
                    self.currstate["eef_pos"][-1] - self.goal_pos[-1]) < 0.01:

                # Check each marker that is still active
                # note that we have an additional marker in the end which we do not want to include
                for site in list(self.sites)[:-1]:
                    # compute distance to site
                    site_pos = self.sites[site].xpos
                    dist = np.linalg.norm(np.array(site_pos) - np.array(self.currstate["eef_pos"]))

                    if dist < 0.025:
                        active_markers.append(site)
        elif self.task_type == "3D":
            # check if we have contact with the object
            if self.total_force_ee >= self.min_force_for_contact:
                # check each marker that is still active
                for site in list(self.sites):
                    site_pos = self.sites[site].xpos
                    dist = np.linalg.norm(np.array(site_pos) - np.array(self.currstate["eef_pos"]))
                    # check if we are close to a site
                    if dist < self.dist_wiped:
                        active_markers.append(site)
        else:
            raise NotImplementedError(f"Expected a known task type of [2D, 3D], but got {self.task_type}...")

        # Obtain the list of currently active (wiped) markers that were not wiped before
        # These are the markers we are wiping at this step
        lall = np.where(np.isin(active_markers, self.wiped_markers, invert=True))  # should only be one index in general
        new_active_markers = np.array(active_markers)[lall]
        # Loop through all new markers we are wiping at this step
        for new_active_marker in new_active_markers:
            # Add this marker the wiped list
            self.wiped_markers.append(new_active_marker)

        if new_active_markers:
            return True
        else:
            return False

    def _update_next_site(self) -> None:
        """
        Updates the next site if the distance between the current site 
        and the robot is under a specific threshold.
        This way the site gets updated even if the site was not wiped 
        according to _update_wiped_markers.
        """
        dist = np.linalg.norm(np.array(self.currstate["eef_pos"]) - np.array(self.site_pos))
        if dist < self.dist_th:
            self.curr_site = next(self.site_cycle)
            self.site_pos = self.sites[self.curr_site].xpos
            self.site_dir = self.sites[self.curr_site].dir

    # Safety features
    def _check_joint_limits(self) -> bool:
        # NOT YET CHECKED AGAINST REAL ROBOT JOINT LIMITS
        """
        Not implemented yet...

        Returns:
            bool: currently passes
        """
        #TODO
        pass

    def _check_collision(self) -> bool:
        """
        Not implemented yet...

        Returns:
            bool: currently passes
        """
        #TODO
        pass

    def _check_cartesian_pos_limits(self) -> bool:
        """
        Check if there is deviation to the trajectory.
        For the 2D task it checks the following: 
            If the current eef-pos lies in a cylinder centered around the goal_pos in the y-direction.
        For the 3D task it checks the following:
            - x_pos around x_pos of goal_pos (smaller than 0.05)
            - y_pos between -0.5 and +0.5
            - z_pos between 0.97 and 1.2
        Returns:
            bool: Whether the cartesian position limits are exceeded or not.
        """
        eef_pos = self.currstate["eef_pos"]
        if self.task_type == "2D":
            self.pos_lim = True if np.linalg.norm(np.array([eef_pos[0] - self.goal_pos[0], eef_pos[2] - self.goal_pos[2]])) > 0.1 \
                       else False
        elif self.task_type == "3D":
            statement1 = np.linalg.norm(np.array(eef_pos[0] - self.goal_pos[0])) > 0.05  
            statement2 = 0.4 < eef_pos[1] or -0.4 > eef_pos[1]
            statement3 = eef_pos[2] < 0.97 or eef_pos[2] > 1.2
            all_statements = statement1 or statement2 or statement3

            self.pos_lim = True if all_statements else False
        return self.pos_lim

    def _check_cartesian_vel_limits(self) -> bool:
        """
        Check for the cartesian velocity via the state (eef_vel_trans) norm and
        verify that the speed lays under the specified threshold defined in the task-configs.
        
        Returns:
            bool: Whether the velocity exceeds the threshold or not.
        """
        statement1 = np.linalg.norm(np.array(self.currstate["eef_vel_trans"])) >= self.vel_cap
        #statement2 = self.currstate["eef_vel_trans"][1] <= -0.02 if self.task_type == "3D" else False  # really important for convergence!
        all_statements = statement1 #or statement2
        
        self.vel_lim = True if all_statements else False
        return self.vel_lim

    def _check_orientation_limits(self) -> bool:
        """
        Checks if the current orientation is within the restricted bounds for the task.

        Returns:
            bool: Whether the orientation exceeds the limits or not.
        """
        curr_ori_euler = R.from_quat(self.currstate["eef_quat"], scalar_first=True).as_euler("xyz", degrees=True)
        statement1 = abs(curr_ori_euler[0] - (180.0)) > 70.0 and abs(curr_ori_euler[0] - (-180.0)) > 70.0
        statement2 = abs(curr_ori_euler[1] - 0.0) > 10.0
        statement3 = abs(curr_ori_euler[2] - 0.0) > 10.0
        all_statements = statement1 or statement2 or statement3

        self.ori_lim = True if all_statements else False
        return self.ori_lim

    def _check_eef_force_limits(self) -> bool:
        """
        Check for the force in 3D and verify that it lays under the specified threshold defined in the task-configs.
        Further, checks that the robot arm has no contact with the table which would lead to rewarding otherwise.

        Returns:
            bool: Whether the force exceeds the threshold or not.
        """
        statement1 = self.total_force_ee >= self.force_cap
        statement2 = (self.currstate["eef_pos"][2] < 1.015) and (np.linalg.norm(np.array(self.currstate["eef_force"]) - np.array(self.ee_force_bias[:3])) >= 1.0)
        all_statements = statement1 or statement2

        self.f_excess = True if all_statements else False
        return self.f_excess
    
    def set_corresponding_state(self, states: dict[str, np.ndarray]) -> None:
        """
        Setter for the associated path points, necessary for the state-space and the rewarding. 
        This function is meant to be called in the prior controller 
        to set the corresponding points according to the current eef position.

        The income will be dictionary of structure 'positions' and 'velocity_dirs' starting with the closest point of the trajectory.
        """
        self.corresponding_path_points = states.get("positions")
        self.corresponding_path_dirs = states.get("velocity_dirs")
        self.corresponding_path_ori = states.get("orientation")

    def _reward_penalty(self, reward_scaled_min=0.0, reward_scaled_max=0.1) -> tuple:
        """
        The reward structure that we started with.
        Including the penalty function from above.
        Scaling the reward in the end. 
        
        Returns:
            float: The reward of the current step.
            dict: The reward info in the form of a dict.
        """
        # helper func
        def scale_reward(reward_val: float) -> float:
            """
            Utility func to scale the reward according to bounds.
            """
            reward_min = self.reward_calc_c_force + self.reward_calc_c_xvel + self.reward_calc_c_xdist + self.reward_calc_c_wvel + self.reward_calc_c_dist_wipe
            reward_max = self.reward_calc_c_contact

            # Scale the reward to desired range
            reward_scaled = (reward_val - reward_min) / (reward_max - reward_min) * np.abs(
                reward_scaled_max - reward_scaled_min) + reward_scaled_min
            
            return reward_scaled

        # Compute force penalty (penalty_force)
        self.episode_forces.append(self.total_force_ee)  # append to episodic forces
        self.penalty_force = _compute_penalty(self.total_force_ee, self.target_force,
                                              self.reward_calc_lower_limit_force, self.reward_calc_upper_limit_force)

        # Compute x velocity penalty (penalty_xvel)
        xvel_ee = self.currstate["eef_vel_trans"][0]
        self.penalty_xvel = _compute_penalty(xvel_ee, self.target_xvel, self.reward_calc_lower_limit_xvel,
                                             self.reward_calc_upper_limit_xvel)

        # Compute x distance penalty (penalty_xdist)
        xdist_ee = self.currstate["eef_pos"][0] - self.goal_pos[0]
        self.penalty_xdist = _compute_penalty(xdist_ee, self.target_xdist, self.reward_calc_lower_limit_xdist,
                                              self.reward_calc_upper_limit_xdist)

        # Compute working velocity direction reward (velocity direction in working direction)
        if self.task_type == "2D":
            wvel_ee = self.currstate["eef_vel_trans"][0]
        elif self.task_type == "3D":
            wvel_ee = np.linalg.norm(np.array(self.currstate["eef_vel_trans"]))
        self.penalty_wvel = _compute_penalty(wvel_ee, self.target_wvel, self.reward_calc_lower_limit_wvel,
                                             self.reward_calc_upper_limit_wvel)
        
        # Compute penalty to current next site point according to the last wiped one
        distance = np.linalg.norm(np.array(self.currstate["eef_pos"]) - np.array(self.site_pos))
        self.penalty_site_dist = _compute_penalty(distance, self.target_dist_wipe, self.reward_calc_lower_limit_dist_wipe, 
                                                  self.reward_calc_upper_limit_dist_wipe) 
        
        # Compute contact reward
        self.reward_contact = 0
        if self.total_force_ee > self.min_force_for_contact:
            self.reward_contact = 1

        reward = self.reward_calc_c_force * self.penalty_force \
                 + self.reward_calc_c_xdist * self.penalty_xdist \
                 + self.reward_calc_c_xvel * self.penalty_xvel \
                 + self.reward_calc_c_wvel * self.penalty_wvel \
                 + self.reward_calc_c_dist_wipe * self.penalty_site_dist \
                 + self.reward_calc_c_contact * self.reward_contact \
        
        reward_scaled = scale_reward(reward_val=reward)

        reward_info = {
            "penalty_force": self.reward_info.get("penalty_force", 0.0) + self.reward_calc_c_force * self.penalty_force,
            "penalty_xdist": self.reward_info.get("penalty_xdist", 0.0) + self.reward_calc_c_xdist * self.penalty_xdist,
            "penalty_xvel": self.reward_info.get("penalty_xvel", 0.0) + self.reward_calc_c_xvel * self.penalty_xvel,
            "penalty_wvel": self.reward_info.get("penalty_wvel", 0.0) + self.reward_calc_c_wvel * self.penalty_wvel,
            "penalty_dist_wipe": self.reward_info.get("penalty_dist_wipe", 0.0) + self.reward_calc_c_dist_wipe * self.penalty_site_dist,
            "reward_contact": self.reward_info.get("reward_contact", 0.0) + self.reward_calc_c_contact * self.reward_contact,
        }
        
        return reward_scaled, reward_info
        

    def _reward_gaussian(self) -> tuple:
        """
        The reward structure for a path following algorithm as recommended by: \
        https://elib.dlr.de/135413/1/2020_Ultsch-et-al_elib_RL-based-PFC-for-a-vehicle-with-variable-delay.pdf \
        Adapted to suit our environment.
        
        Returns:
            float: The reward of the current step.
            dict: The reward info in the form of a dict.
        """
        # Compute force penalty (penalty_force)
        self.episode_forces.append(self.total_force_ee)  # append to episodic forces
        reward_force_error = _compute_gaussian_reward(self.total_force_ee - self.target_force, theta=self.theta_force)

        # PATH REWARD ====================
        # Compute cross error with projection to trajectory direction
        deviation_for_trajectory = self.corresponding_path_points[0] - self.currstate["eef_pos"]
        cross_error = np.linalg.norm(deviation_for_trajectory - np.dot(deviation_for_trajectory,self.corresponding_path_dirs[0])*self.corresponding_path_dirs[0])
        reward_cross_error = _compute_gaussian_reward(value=cross_error, theta=self.theta_cross)

        # Compute velocity direction error
        curr_dir = self.currstate["eef_pos"] - self.last_point
        direction_error = np.arccos(np.dot(curr_dir,self.corresponding_path_dirs[0])/(np.linalg.norm(curr_dir)*np.linalg.norm(self.corresponding_path_dirs[0])))
        reward_direction_error = _compute_gaussian_reward(value=direction_error, theta=self.theta_dir)

        # Compute absolute velocity error
        abs_vel = np.linalg.norm(np.array(self.currstate["eef_vel_trans"]))
        abs_vel_error = abs_vel - self.target_wvel
        reward_abs_vel_error = _compute_gaussian_reward(value=abs_vel_error, theta=self.theta_vel)
        # ==================== PATH REWARD

        reward = reward_cross_error * (1 + reward_direction_error + reward_abs_vel_error) + reward_force_error
                 
        reward_info = {
            "reward_cross_error": self.reward_info.get("reward_cross_error", 0.0) + reward_cross_error/self.theta_cross[0],
            "reward_dir_error": self.reward_info.get("reward_dir_error", 0.0) + reward_direction_error/self.theta_dir[0],
            "reward_abs_vel_error": self.reward_info.get("reward_abs_vel_error", 0.0) + reward_abs_vel_error/self.theta_vel[0],
            "reward_force_error": self.reward_info.get("reward_force_error", 0.0) + reward_force_error/self.theta_force[0],
        }
        
        return reward, reward_info
    
    def _reward_linear(self) -> tuple:
        """
        The reward structure adapted from _reward_gaussian but with remarks from Emma.

        Returns:
            float: The reward of the current step.
            dict: The reward info in the form of a dict.
        """

        norm_calc_c_force, norm_calc_c_cross, norm_calc_c_dir, norm_calc_c_vel = _normalize_weighting(self.reward_calc_c_force, self.reward_calc_c_cross, self.reward_calc_c_dir, self.reward_calc_c_vel_error)

        # Compute force reward
        self.episode_forces.append(self.total_force_ee)  # append to episodic forces
        reward_force_error = norm_calc_c_force * _compute_linear_reward(self.total_force_ee, self.target_force, self.reward_calc_lower_limit_force, self.reward_calc_upper_limit_force)

        # PATH REWARD ====================
        # Compute cross error with projection to trajectory direction
        deviation_for_trajectory = self.corresponding_path_points[0] - self.currstate["eef_pos"]
        cross_error = np.linalg.norm(deviation_for_trajectory)
        reward_cross_error = norm_calc_c_cross * _compute_linear_reward(cross_error, self.target_cross_error, self.reward_calc_lower_limit_cross, self.reward_calc_upper_limit_cross)

        # Compute velocity direction error
        curr_dir = self.currstate["eef_pos"] - self.last_point
        direction_error = np.arccos(np.dot(curr_dir,self.corresponding_path_dirs[0])/(np.linalg.norm(curr_dir)*np.linalg.norm(self.corresponding_path_dirs[0])))
        reward_direction_error = norm_calc_c_dir * _compute_linear_reward(direction_error, self.target_dir_error, self.reward_calc_lower_limit_dir, self.reward_calc_upper_limit_dir)

        # Compute absolute velocity error
        abs_vel = np.linalg.norm(np.array(self.currstate["eef_vel_trans"]))
        abs_vel_error = abs(abs_vel - self.target_wvel)
        reward_abs_vel_error = norm_calc_c_vel * _compute_linear_reward(abs_vel_error, self.target_vel_error, self.reward_calc_lower_limit_vel_error, self.reward_calc_upper_limit_vel_error)
        # ==================== PATH REWARD
        
        reward = reward_cross_error + reward_abs_vel_error + reward_force_error + reward_direction_error
                 
        reward_info = {
            "reward_cross_error": self.reward_info.get("reward_cross_error", 0.0) + reward_cross_error/norm_calc_c_cross,
            "reward_abs_vel_error": self.reward_info.get("reward_abs_vel_error", 0.0) + reward_abs_vel_error/norm_calc_c_vel,
            "reward_dir_error": self.reward_info.get("reward_dir_error", 0.0) + reward_direction_error/norm_calc_c_dir,
            "reward_force_error": self.reward_info.get("reward_force_error", 0.0) + reward_force_error/norm_calc_c_force,
        }

        return reward, reward_info

    def _reward_real_robot(self) -> tuple:
        """
        The reward function that we find to be working for the real robot.

        Returns:
            float: The reward of the current step.
            dict: The reward info in the form of a dict.
        """
        norm_calc_c_force, norm_calc_c_cross, norm_calc_c_dir, norm_calc_c_vel = _normalize_weighting(self.reward_calc_c_force, self.reward_calc_c_cross, self.reward_calc_c_dir, self.reward_calc_c_vel_error)

        # Compute force reward
        self.episode_forces.append(self.total_force_ee)  # append to episodic forces
        reward_force_error = norm_calc_c_force * _compute_piecewise_linear_quad_reward(self.total_force_ee, self.target_force,
                                                                                       self.reward_calc_lower_limit_force_outer,
                                                                                       self.reward_calc_lower_limit_force_inner,
                                                                                       self.reward_calc_upper_limit_force_inner,
                                                                                       self.reward_calc_upper_limit_force_outer)

        # PATH REWARD ====================
        # Compute cross error with projection to trajectory direction
        deviation_for_trajectory_x = self.corresponding_path_points[0][0] - self.currstate["eef_pos"][0]
        cross_error_x = np.linalg.norm(deviation_for_trajectory_x)
        reward_cross_error_x = 1/3 * norm_calc_c_cross * _compute_linear_reward(cross_error_x, self.target_cross_error,
                                                                                self.reward_calc_lower_limit_cross,
                                                                                self.reward_calc_upper_limit_cross)
        deviation_for_trajectory_yz = self.corresponding_path_points[0][1:] - self.currstate["eef_pos"][1:]
        cross_error_yz = np.linalg.norm(deviation_for_trajectory_yz)
        reward_cross_error_yz = 2/3 * norm_calc_c_cross * _compute_linear_reward(cross_error_yz, self.target_cross_error,
                                                                                 self.reward_calc_lower_limit_cross,
                                                                                 self.reward_calc_upper_limit_cross)

        # Compute velocity direction error
        curr_dir = self.currstate["eef_pos"] - self.last_point
        try:
            unit_curr_dir = curr_dir/np.linalg.norm(curr_dir)
            direction_error = np.arccos(np.dot(unit_curr_dir,self.corresponding_path_dirs[0])/np.linalg.norm(self.corresponding_path_dirs[0]))
            reward_direction_error = norm_calc_c_dir * _compute_linear_reward(direction_error, self.target_dir_error, 
                                                                              self.reward_calc_lower_limit_dir, 
                                                                              self.reward_calc_upper_limit_dir)
        except (RuntimeWarning, RuntimeError):
            reward_direction_error = 0.0
        if np.linalg.norm(np.array(self.currstate["eef_vel_trans"])) < 0.015:
            # in case velocity is low -> get rid of reward for direction error
            reward_direction_error = 0.0

        # Compute absolute velocity error projected on direction of movement
        abs_vel = np.linalg.norm(np.array(self.currstate["eef_vel_trans"]))
        abs_vel_error = abs_vel - self.target_wvel
        reward_abs_vel_error = norm_calc_c_vel * _compute_linear_reward(abs_vel_error, self.target_vel_error,
                                                                        self.reward_calc_lower_limit_vel_error,
                                                                        self.reward_calc_upper_limit_vel_error)
        # ==================== PATH REWARD

        reward = reward_cross_error_x + reward_cross_error_yz + reward_abs_vel_error + reward_force_error + reward_direction_error

        reward_info = {
            "reward_cross_x_error": self.reward_info.get("reward_cross_x_error", 0.0) + reward_cross_error_x / (1/3 * norm_calc_c_cross),
            "reward_cross_yz_error": self.reward_info.get("reward_cross_yz_error", 0.0) + reward_cross_error_yz / (2/3 * norm_calc_c_cross),
            "reward_abs_vel_error": self.reward_info.get("reward_abs_vel_error", 0.0) + reward_abs_vel_error / norm_calc_c_vel,
            "reward_dir_error": self.reward_info.get("reward_dir_error", 0.0) + reward_direction_error / norm_calc_c_dir,
            "reward_force_error": self.reward_info.get("reward_force_error", 0.0) + reward_force_error / norm_calc_c_force,
        }
        return reward, reward_info

    
    def _median_force_computation(self) -> np.ndarray:
        """
        Computes the median of the force buffer, collected after each simulation step.

        Returns:
            np.ndarray: The 3D-force value representing the median of the absolute force values.
        """
        try:
            numerated_force_buffer = {key: value for (key, value) in enumerate(self.force_buffer)}
            sorted_force_buffer = {key: value for (key, value) in sorted(numerated_force_buffer.items(), key=lambda item: np.linalg.norm(item[1]))}
            
            median_force = list(sorted_force_buffer.items())[len(sorted_force_buffer) // 2][1]
            self.total_force_ee = np.linalg.norm(median_force)
        except:
            median_force = np.zeros(3)
            self.total_force_ee = 0

        return median_force
