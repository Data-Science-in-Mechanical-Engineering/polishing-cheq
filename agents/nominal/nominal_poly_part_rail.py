import torch
import numpy as np
import gymnasium as gym
from typing import Type, Union
from collections import OrderedDict
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from agents.nominal.nominal_base import NominalBase


class NominalPolyPartRail(NominalBase):
    """
    This class is used to create a controller for the SimulationRobot.
    It uses polynomial interpolation to create a trajectory based on via_points.
    Further, for the orientation of the end-effector it uses SLERP as an interpolation method for angular motion.
    Thus, after initialization, the full trajectory is achieved.
    
    It further includes a control law for the get_action method that takes the current state
    of the robot as base for the desired point computation.
    This will come in handy for the MixedAgent.

    In addition to the NominalPolyRail, this controller includes a change in impedance gains 
    which is based on passing a certain position of the y-axis.

    Args:
        NominalBase (class): The general parent of controls
    """

    def __init__(     
            self, 
            env: Type[gym.Wrapper],
            via_points: dict,
            spacing: float,
            resolution: int,
            nominal_gains: list[list[float]],
            bounds_change: list[float],
            indent: float,
            action_radius: float,
            num_skip_points: int,
            device: torch.device = "cpu",
    ):
        """
        Initialization of the nominal agent based on the provided environment.
        """
        super().__init__(env)
        self.env = env
        self.via_points = via_points
        self.spacing = spacing
        self.resolution = resolution
        self.nominal_gains = np.array(nominal_gains)
        self.bounds_change = bounds_change
        self.indent = indent
        self.action_radius = action_radius
        self.num_skip_points = num_skip_points if num_skip_points is not None else 0
        self.device = device

        assert len(bounds_change) == len(nominal_gains)-1,  \
            "Please specify correct nominal gains and change positions in y-direction."
        
        # initialize trajectory and discretization
        self.num_sections = len(self.via_points.keys())-1
        self.save_orientation_key_points = []

        self.trajectory = self._compute_polynomials()
        self.control_points = self._calculate_points()
        self.orientation_points = self._calculate_orientation()

        print(f"Nominal controller is defined with {self.control_points.shape[0]} points.")

        assert len(self.control_points) == len(self.orientation_points), \
            f"Expects the to have the same number of control points and orientation points, but got {len(self.control_points)} and {len(self.orientation_points)}."

        # for action control:
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
        
        # receive current state from the observation ==========================
        curr_quat = np.array(obs[31:35])
        current_rotation = R.from_quat(curr_quat, scalar_first=True)
        z_curr = current_rotation.apply(np.array([0, 0, 1]))
        self.indent_rot = self.indent * z_curr
        eef_pos = np.array(obs[28:31]) + self.indent_rot
        # ========================== receive current state from the observation

        # compute next control point ==========================================
        close_points = OrderedDict()
        overall_dist = []
        for i, point in enumerate(self.control_points):
            # iterate over all created control points (possible IMPROVEMENT to take subset)
            dist = np.linalg.norm(eef_pos - point)
            overall_dist.append(dist)
            if dist <= self.action_radius:
                close_points[i] = point
            else:
                pass
        
        # send corresponding trajectory points to environment for rewarding and state
        closest_point_index = np.argmin(overall_dist)
        self._send_points_to_env(point_index=closest_point_index)

        if len(close_points.keys()) == 0:
            # case: we have no point in action_radius
            point_index = closest_point_index
            # further increase point_index by num_skip_points for 
            point_index += self.num_skip_points if point_index+self.num_skip_points <= len(self.control_points)-1 else 0

        else:
            # case: there are points in the action radius
            point_index = list(close_points.keys())[-1]  # take the last index of these points
        
        del close_points, overall_dist, closest_point_index

        # check how high the velocity of the controller currently is
        vel_trans = np.array(obs)[35:38]
        """if (np.linalg.norm(vel_trans) < 0.02):
            point_index += int(self.num_skip_points*1.5) if int(point_index+1.5*self.num_skip_points) <= len(self.control_points)-1 else 0
            self.next_pos = self.control_points[point_index]
            pos_action = self.next_pos - eef_pos
        elif(np.linalg.norm(vel_trans) < 0.02):
            self.next_pos = self.control_points[point_index]
            pos_action = self.next_pos - eef_pos
            pos_action = self._enhance_abs_action(pos_action, 1.5)
        else:"""
        self.next_pos = self.control_points[point_index]
        pos_action = self.next_pos - eef_pos
        # ========================================== compute next control point

        # adaptive gains based on position and defined bounds_change
        gains_index = 0
        for ind, bound_pos in enumerate(self.bounds_change):
            if obs[29] < bound_pos:
                gains_index = ind + 1
        action[-7:] = torch.tensor(self.nominal_gains[gains_index])

        # computed positional shift
        action[:3] = torch.tensor(pos_action)

        # orientational shift
        if self.env.unwrapped.control_mode == "pose":
            self.next_quat = self.orientation_points[point_index]  # the one we like to have
            trans = R.from_quat(self.next_quat, scalar_first=True) * R.from_quat(curr_quat, scalar_first=True).inv()  # computation of necessary transformation
            action[3:6] = torch.tensor(trans.as_euler("xyz", degrees=True))  # give transformation as euler angle
        else:
            pass
        
        # scale and clip action
        action = self.env.unwrapped.scaling_func(action)
        action = torch.clip(action, torch.tensor(self.env.unwrapped.action_space.low), torch.tensor(self.env.unwrapped.action_space.high))
        
        return action.to(device=self.device)


    def _calculate_points(self) -> np.ndarray:
        """
        Based on the trajectory polynomials specified in self.trajectory,
        we can compute some points in the trajectory. These will be used as 
        control points for the actions.
        For this we are using equidistant control points since this will help us 
        with the constant velocity.

        Returns:
            np.ndarray: in the shape of N x 3 (3-dimensional)
        """
        # initialization
        points = []
        last_control_point = self._get_func_value(t=0, coef_dict=self.trajectory)
        current_via_pos = 0.0

        # compute equidistant control points 
        for t in np.linspace(start=0, stop=self.num_sections, num=int(self.num_sections/self.resolution + 1)):
            curr_control_point = self._get_func_value(t=t, coef_dict=self.trajectory)

            dist = np.linalg.norm(last_control_point - curr_control_point)
            if dist < self.spacing:
                pass  # if distance is under specified threshold
            else:
                points.append(curr_control_point)
                last_control_point = curr_control_point
            
            if abs(t-current_via_pos) <= self.resolution:  # note that this is not exactly correct here, but solves the purpose
                # save point index where we come across a via point for the orientation computation
                self.save_orientation_key_points.append(len(points))
                current_via_pos += 1.0

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
            key_times.append(self.save_orientation_key_points[i])

        # i might have to adapt the key_times
        slerp = Slerp(times=key_times, rotations=R.from_euler(seq="xyz", angles=np.array(via_oris), degrees=True))
        orientation_points = slerp(range(0,self.control_points.shape[0])).as_quat(scalar_first=True)  # the same number as control points

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
    
    @staticmethod
    def _get_func_value(t: float, coef_dict: dict) -> Union[np.ndarray, None]:
        """
        Receive the trajectory point of a specific trajectory give by coef_dict when specifying t

        Args:
            t (float): The point to evaltuate the trajectory
            coef_dict (dict): The dictionary specifying the trajectory

        Returns:
            Union[np.ndarray, None]: Either returns the 3D point or None if t is not in the trajectory bounds.
        """
        num_sections = len(coef_dict)
        for section_dict in coef_dict.values():
            if (section_dict["bound_min"] <= t < section_dict["bound_max"]) or (t == num_sections and t == section_dict["bound_max"]):
                # last condition for last step 
                x_func = lambda t, c=section_dict["x_traj_coeffs"], time_min=section_dict["bound_min"], time_max=section_dict["bound_max"]: c[0] + (t-time_min)/(time_max-time_min) * c[1] + ((t-time_min)/(time_max-time_min))**2 * c[2] + ((t-time_min)/(time_max-time_min))**3 * c[3]
                y_func = lambda t, c=section_dict["y_traj_coeffs"], time_min=section_dict["bound_min"], time_max=section_dict["bound_max"]: c[0] + (t-time_min)/(time_max-time_min) * c[1] + ((t-time_min)/(time_max-time_min))**2 * c[2] + ((t-time_min)/(time_max-time_min))**3 * c[3]
                z_func = lambda t, c=section_dict["z_traj_coeffs"], time_min=section_dict["bound_min"], time_max=section_dict["bound_max"]: c[0] + (t-time_min)/(time_max-time_min) * c[1] + ((t-time_min)/(time_max-time_min))**2 * c[2] + ((t-time_min)/(time_max-time_min))**3 * c[3]    
                return np.array([x_func(t), y_func(t), z_func(t)])
            else:
                pass

        return None
    
    def _send_points_to_env(self, point_index: int, num_points: int=5) -> None:
        """
        This function sends the next points of the trajectory to the environment.
        These are needed for rewarding and state shaping.
        All based on the current position of the end-effector.

        Args:
            next_point_index (int): the index to the current next point of the trajectory
            num_points (int): how many next points are send
        
        Details:
            If the next num_points do not exist in the trajectory, takes the last direction (end_dir) of the trajectory
            and artificially extend the trajectory following the end_dir.
            The directions created are based on the difference to the next point.
            In order to get the real trajectory point we add the indent to the position.
            Also adds the current orientation of the next point as a key to the directory.
        """
        end_dir = (self.control_points[-1] - self.control_points[-2])/np.linalg.norm(self.control_points[-1] - self.control_points[-2])

        positions, directions = [], []
        for running_index in range(0, num_points):
            if point_index+running_index < len(self.control_points):
                # computation of rotated indentation
                current_rotation = R.from_quat(self.orientation_points[point_index], scalar_first=True)
                z_new = current_rotation.apply(np.array([0, 0, 1]))
                indent_rot = self.indent * z_new

                # next point does exist
                current_point = self.control_points[point_index+running_index]
                positions.append(current_point + indent_rot)
                if point_index+running_index+1 < len(self.control_points):
                    # next point after next point does exist
                    next_point = self.control_points[point_index+running_index+1]
                    directions.append((next_point - current_point)/np.linalg.norm(next_point - current_point))
                else:
                    # next point after next point does not exist
                    directions.append(end_dir)
            else:
                # next point does not exist
                positions.append(positions[-1] + end_dir + indent_rot)  # take the last computed indent_rot here
                directions.append(end_dir)
        
        corresponding_states = {
            "positions": np.array(positions), 
            "velocity_dirs": np.array(directions),
            "orientation": R.from_quat(self.orientation_points[point_index], scalar_first=True).as_euler("xyz", degrees=True)
        }
        self.env.unwrapped.set_corresponding_state(states=corresponding_states)

    @staticmethod
    def _enhance_abs_action(action: np.ndarray, multiplier: float) -> np.ndarray:
        return multiplier * action
    
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
    