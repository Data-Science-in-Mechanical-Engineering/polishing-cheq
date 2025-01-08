import torch
import numpy as np
import gymnasium as gym
from typing import Type
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from agents.nominal.nominal_base import NominalBase


class NominalPolyBasic(NominalBase):
    """
    This class is used to create a controller for the SimulationRobot.
    It uses polynomial interpolation to create a trajectory based on via_points.
    Further, for the orientation of the end-effector it uses SLERP as an interpolation method for angular motion.

    Thus, after initialization, the full trajectory is achieved.
    The controlling is then just acting on the sampling points of the trajectory created by
    even distributed time sections.

    Args:
        NominalBase (class): The general parent of controls
    """

    def __init__(     
            self, 
            env: Type[gym.Wrapper],
            via_points: dict,
            num_points: int,
            nominal_gains: list[float],
            indent: float,
            device: torch.device = "cpu",
    ):
        """
        Initialization of the nominal agent based on the provided environment.
        """
        super().__init__(env)
        self.env = env
        self.via_points = via_points
        self.num_points = num_points
        self.nominal_gains = nominal_gains
        self.indent = indent
        self.device = device
        
        # initialize discretization
        self.num_sections = len(self.via_points.keys())-1

        # receive trajectory specifics and control points
        self.traj_steps = np.linspace(start=0.0, stop=self.num_sections, num=self.num_points)
        self.trajectory = self._compute_polynomials()
        self.control_points = self._calculate_points()
        self.orientation_points = self._calculate_orientation()

        assert len(self.control_points) == len(self.orientation_points), \
            f"Expects the to have the same number of control points and orientation points, but got {len(self.control_points)} and {len(self.orientation_points)}."

        # for action control:
        self.point_index = 0
        self.next_pos = None
        self.next_quat = None
        self.indent_rot = None

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Computes the nominal action based on the provided observation

        Args:
            obs (torch.Tensor): The current observation / state of the robot

        Returns:
            torch.Tensor: The next action to perform.
        """
        action = torch.zeros(self.env.unwrapped.action_space.shape)

        eef_pos = obs[28:31]  # assuming that we find the eef_pos here
        self.next_pos = self.control_points[self.point_index]
        action[-7:] = torch.tensor(self.nominal_gains)
        action[:3] = torch.tensor(self.next_pos) - eef_pos[:3]

        # orientational shift
        if self.env.unwrapped.control_mode == "pose":
            curr_quat = obs[31:35]  # current quaternion
            self.next_quat = self.orientation_points[self.point_index]  # the one we like to have
            trans = R.from_quat(self.next_quat) * R.from_quat(curr_quat).inv()  # computation of necessary transformation
            action[3:6] = torch.tensor(trans.as_euler("xyz", degrees=True))  # give transformation as euler angle
        else:
            pass

        # further perform subtraction of indent in dependence of current head orientation
        current_rotation = R.from_euler("x", R.from_quat(self.next_quat).as_euler("xyz", degrees=True)[-1], degrees=True)
        z_new = current_rotation.apply(np.array([0, 0, 1]))
        indent_rot = self.indent * z_new
        self.indent_rot = indent_rot
        action[:3] += torch.tensor(indent_rot)
        
        action = torch.clip(action, torch.tensor(self.env.unwrapped.action_space.low), torch.tensor(self.env.unwrapped.action_space.high))

        self._update_point_index()
        
        return action.to(device=self.device)

    def _calculate_points(self) -> np.ndarray:
        """
        Based on the trajectory polynomials specified in self.trajectory,
        we can compute some points in the trajectory. These will be used as 
        control points for the actions.

        Returns:
            np.ndarray: in the shape of N x 3 (3-dimensional)
        """
        # initialization
        points = []

        for step in self.traj_steps:
            for section_dict in self.trajectory.values():
                if (section_dict["bound_min"] <= step < section_dict["bound_max"]) or (step == self.num_sections and step == section_dict["bound_max"]):
                    # last condition for last step 
                    x_func = lambda t, c=section_dict["x_traj_coeffs"], time_min=section_dict["bound_min"], time_max=section_dict["bound_max"]: c[0] + (t-time_min)/(time_max-time_min) * c[1] + ((t-time_min)/(time_max-time_min))**2 * c[2] + ((t-time_min)/(time_max-time_min))**3 * c[3]
                    y_func = lambda t, c=section_dict["y_traj_coeffs"], time_min=section_dict["bound_min"], time_max=section_dict["bound_max"]: c[0] + (t-time_min)/(time_max-time_min) * c[1] + ((t-time_min)/(time_max-time_min))**2 * c[2] + ((t-time_min)/(time_max-time_min))**3 * c[3]
                    z_func = lambda t, c=section_dict["z_traj_coeffs"], time_min=section_dict["bound_min"], time_max=section_dict["bound_max"]: c[0] + (t-time_min)/(time_max-time_min) * c[1] + ((t-time_min)/(time_max-time_min))**2 * c[2] + ((t-time_min)/(time_max-time_min))**3 * c[3]
                else:
                    pass

            points.append([x_func(step), y_func(step), z_func(step)])

        return np.array(points)
    
    def _calculate_orientation(self) -> np.ndarray:
        """
        Based on the SLERP algorithm and the specification of the via points, compute the orientation trajectory.

        Returns:
            np.ndarray: The orientation trajectory points
        """
        via_oris = []
        key_times = []

        for i in range(self.num_sections+1):  # since we want to grab the last one as well
            # grab via points specifications for the specific sections
            ori = self.via_points[f"point_{str(i)}"]["ori"]
            via_oris.append(ori)
            key_times.append(i)

        # i might have to adapt the key_times
        slerp = Slerp(times=key_times, rotations=R.from_euler(seq="xyz", angles=np.array(via_oris), degrees=True))
        orientation_points = slerp(self.traj_steps).as_quat()  # the same number as control points

        return orientation_points

    def _compute_polynomials(self) -> dict:
        """
        Generates the polynomial coefficients for the trajectory
        based on the via-points specifications.

        Returns:
            dict: The trajectory specifics in the form of dict(key, dict(coefficients))
        """
        coef_dict = dict()

        for i in range(self.num_sections):
            # grab via points specifications for specific sections
            start_pos = self.via_points[f"point_{str(i)}"]["pos"]
            start_vel = self.via_points[f"point_{str(i)}"]["vel"]

            end_pos = self.via_points[f"point_{str(i+1)}"]["pos"]
            end_vel = self.via_points[f"point_{str(i+1)}"]["vel"]

            x_traj = self._compute_coefficients(
                conditions=[start_pos[0], start_vel[0], end_pos[0], end_vel[0]],
                time_start=float(i), time_end=float(i+1))
            y_traj = self._compute_coefficients(
                conditions=[start_pos[1], start_vel[1], end_pos[1], end_vel[1]],
                time_start=float(i), time_end=float(i+1))
            z_traj = self._compute_coefficients(
                conditions=[start_pos[2], start_vel[2], end_pos[2], end_vel[2]],
                time_start=float(i), time_end=float(i+1))
            
            current_dict = {
                "x_traj_coeffs": x_traj,
                "y_traj_coeffs": y_traj,
                "z_traj_coeffs": z_traj,
                "bound_min": float(i),
                "bound_max": float(i+1),
            }
            coef_dict[i] = current_dict

        return coef_dict

    @staticmethod
    def _compute_coefficients(conditions: list, time_start: float, time_end: float) -> tuple:
        """
        The computation of the coefficients is based on the start and the beginning conditions of the trajectory section.
        These conditions have to be given in the form of the constraints list. Which should look like this:
        conditions: [start_pos, start_vel, end_pos, end_vel]

        Args:
            conditions (list): The list of the conditions for the trajectory section
            time_start (float): Start time for the trajectory
            time_end (float): End time for the trajectory

        Returns:
            tuple: The four coefficients for the polynomial trajectory
        """
        assert len(conditions)==4, "You should have 4 constraints!"

        a_j0 = conditions[0]
        a_j1 = conditions[1]
        a_j2 = (3*conditions[2] - 3*conditions[0] - 2*conditions[1]*(time_end-time_start) - conditions[3]*(time_end-time_start)) / (time_end - time_start)**2
        a_j3 = (2*conditions[0] + (conditions[1] + conditions[3])*(time_end-time_start) - 2*conditions[2]) / (time_end-time_start)**3

        return (a_j0, a_j1, a_j2, a_j3)

    def _update_point_index(self):
        """
        Checks if the environment has been reset by verifying that the environment_step is back at 0.
        In this case, reset the point_index to 0 again.
        Otherwise updates point_index by one step.
        """
        if self.env.unwrapped.environment_step == 0:
            self.point_index = 0
        else:
            self.point_index += 1
    

    @property
    def get_next_pos(self) -> np.ndarray:
        """
        Getter for the next set cartesian position in 3D.
        """
        return self.next_pos
    
    @property
    def get_next_quat(self) -> np.ndarray:
        """
        Getter for the next set quaternion for the orientation.
        """
        return self.next_quat
    
    @property
    def get_trajectory_points(self) -> np.ndarray:
        """
        Getter for the trajectory points after the creation in the init.
        """
        return self.control_points
    
    @property
    def get_orientation_points(self) -> np.ndarray:
        """
        Getter for the orientation points of the trajectory after the creation in the init.
        """
        return self.orientation_points
    
    @property
    def get_curr_indent(self) -> np.ndarray:
        """
        Getter for the current indentation.
        """
        return self.indent_rot
    