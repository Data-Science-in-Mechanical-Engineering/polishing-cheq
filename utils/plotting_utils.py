import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import torch
import numpy as np
import pandas as pd
import os

from wandb import Api
from typing import Literal, Union, Optional, List
from tqdm.notebook import tqdm
from pathlib import Path


def plot_force_along_path(data):
    '''
    This function plots the force along the 2d projected path.

    :param data: A pandas dataframe with the following columns: 'x_pos', 'y_pos', 'force'.

    :return: plt.figure: A figure object containing the plot.

    '''
    # extract relevant data from dataframe
    x = data['x_pos'].to_numpy()
    y = data['y_pos'].to_numpy()
    force = data['force'].to_numpy()

    fig, ax = plt.subplots()

    # Normalizing the 'force' values for color mapping
    norm = plt.Normalize(force.min(), force.max())

    # Creating a colormap
    colormap = plt.get_cmap('viridis')

    # Plot each segment with color based on the 'force' value
    for i in range(len(x) - 1):
        ax.plot(y[i:i+2], x[i:i+2], color=colormap(norm(force[i:i+2].mean())))

    # Setting labels
    ax.set_xlabel('Y Position [m]')
    ax.set_ylabel('X Position [m]')

    # Adding color bar to indicate force values
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    mappable.set_array(force.astype(float))
    fig.colorbar(mappable, ax=ax, label='Force [N]')

    return fig

def save_figure_in_docs(
        figure: Figure, 
        filename: str, 
        path_dir: str = "../docs/figures_res/", 
        format: Literal["png", "pdf"] = None, 
        **kwargs
    ) -> None:
    """
    Utility function to save a figure in the docs directory.
    Args:
        figure (Figure): Figure to be saved
        filename (str): specification of filename
        path_dir (Optional[str]): path directory. Defaults to "../docs/figures_res/"
        format (Literal[&quot;png&quot;, &quot;pdf&quot;], optional): Format-type. Defaults to None.
    """
    if format is None:
        assert len(filename.split("/")[-1].split(".")) > 1, "Please specify a format type in the filename or in the format-kwarg."
    if not os.path.isdir(path_dir):
        raise Exception(f"Please specify a suitable path directory for the figure saving or create the directory. Got {path_dir}")
    figure.savefig(path_dir+filename, format=format, bbox_inches="tight", **kwargs)

def receive_data_from_api(
        api: Api, 
        steps: int, 
        run_paths: dict, 
        vals: List[str], 
        buffer_size: int = 100000
    ) -> List[float]:
    """
    Grabs data from the WandB Api and creates a list as output.
    (Assumes to be called in a notebook.)

    Args:
        api (Api): The used WandB api.
        steps (int): The maximal step to of the learning to consider
        run_paths (dict): The path to the experiments
        vals (list): The values to receive from the experiments
        buffer_size (Optional): The size of received data at a time (defaults to 100.000).

    Returns: 
        list: Nested List with data from wandb
    """
    data = None
    pbar = tqdm(range(0, steps, buffer_size))
    for step in pbar:
        pbar.set_description(f"Processing for steps {step} - {step+buffer_size}")
        runs = [api.run(run_p) for run_p in run_paths.values()]
        metrics = [run.scan_history(page_size=buffer_size, min_step=step, max_step=step + buffer_size) for run in runs]
        data_new = [[[row[val] for val in vals] for row in metric] for metric in metrics]
        if data is None:
            data = data_new
        else:
            for run_id, run in enumerate(data_new):
                data[run_id] += run

    return data

def export_legend(legend, filename: str, expand=[-1, -1, 1, 1]):
    """Exports a legend to a file.
    Useful for showing plots in latex.

    Args:
        legend (Legend): the legend to save
        filename (str): Name of the file (+fileformat).
        expand (list, optional): padding around legend. Defaults to [-1, -1, 1, 1].
    """
    fig_exp  = legend.figure
    fig_exp.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig_exp.dpi_scale_trans.inverted())
    fig_exp.savefig(filename, dpi="figure", bbox_inches=bbox)
    

class EvalObserver:
    """
    Powerful tool to observe the evaluation runs of any kind.
    Also enables to save the data to a csv file and reload it when in need.
    """
    STIFF_INDEX_DICT = {"x": 0, "y": 1, "z": 2, "rot_x": 3, "rot_y": 4, "rot_z": 5}

    def __init__(self, load: bool = False):
        """
        Initilialize EvaluationObserver by creating empty dictionaries.
        """
        self.load = load
        self.num_runs = 0

        self._states = dict()
        
        self._hybrid_actions = dict()
        self._rl_actions = dict()
        self._nominal_actions = dict()

        self._forces = dict()
        self._steps = dict()

        self._reward_force_error = dict()
        self._reward_cross_x_error = dict()
        self._reward_cross_yz_error = dict()
        self._reward_abs_vel_error = dict()
        self._reward_dir_error = dict()        


    ########## setter-methods ##########

    def register_state(self, obs: torch.Tensor, iteration: int) -> None:
        """
        Register a state to the EvalObserver.

        Args:
            obs (torch.Tensor): state to add
            iteration (int): which evaluation iteration are we in

        Note:
            This is the dictionary which is used to determine the number of runs
        """
        if isinstance(obs, torch.Tensor):
            obs = np.array(obs.cpu())
        else:
            pass

        try:
            self._states[iteration].append(obs)
        except KeyError:
            self._states[iteration] = [obs]
            self.num_runs += 1
        except Exception as e:
            raise Exception(e)
    
    def register_action(self, action: Union[np.ndarray,torch.Tensor], iteration: int, key: Literal["hybrid", "rl", "nominal"]) -> None:
        """
        Register an action to the EvalObserver.

        Args:
            action (Union[np.ndarray,torch.Tensor]): action to add, depending on where it is from, it might be array or tensor.
            iteration (int): which evaluation iteration are we in
        """
        assert isinstance(action, torch.Tensor) or isinstance(action, np.ndarray), "Please insert an action of type numpy or tensor."
        
        if isinstance(action, torch.Tensor):
            action = np.array(action.cpu())
        else:
            pass
        
        if key == "hybrid":
            try:
                self._hybrid_actions[iteration].append(action)
            except KeyError:
                self._hybrid_actions[iteration] = [action]
        elif key == "rl":
            try:
                self._rl_actions[iteration].append(action)
            except KeyError:
                self._rl_actions[iteration] = [action]
        elif key == "nominal":
            try:
                self._nominal_actions[iteration].append(action)
            except KeyError:
                self._nominal_actions[iteration] = [action]
        
    def register_step(self, step: int, iteration: int) -> None:
        """
        Register a step to the EvalObserver.

        Args:
            step (int): The episodic step to add
            iteration (int): which evaluation iteration are we in
        """
        try: 
            self._steps[iteration].append(step)
        except KeyError:
            self._steps[iteration] = [step]
        except Exception as e:
            raise Exception(e)
        
    def register_info(self, info: dict, iteration: int) -> None:
        """
        Registers the info of the step and extracts the force.

        Args:
            info (dict): The info of the current step
            iteration (int): which evaluation iteration are we in
        """
        try:
            self._forces[iteration].append(info["force"])
        except KeyError:
            self._forces[iteration] = [info["force"]]

    def register_reward(self, reward_info: dict, iteration: int) -> None:
        """Register reward info and sorts their parts according to the key specified in the info-dict.
        """
        for key, value in reward_info.items():
            try:
                attr: dict = getattr(self, f"_{key}")
            except AttributeError:
                raise AttributeError(f"EvalObserver assumed to have attribute self._{key}...")
            try:
                attr[iteration].append(value)
            except KeyError:
                attr[iteration] = [value]


    ########## getter-methods ##########
        
    def get_steps(self, runs: Optional[List[int]] = None) -> np.ndarray:
        """
        Returns steps of the runs.
        If runs is not specified, returns the complete list of runs.
        To receive same lengths of the runs, all arrays are cut at minimal length of all runs chosen.
        """
        if runs is None:
            runs = range(1,self.num_runs+1)

        min_len = min([len(self._steps[r]) for r in runs])
        return np.array([np.array(self._steps[r][:min_len-1]) for r in runs])
    
    def get_3D_pos(self, runs: Optional[List[int]] = None) -> np.ndarray:
        """
        Returns 3D position of the run.
        """
        return np.array(self._states[runs])[:, 28:31]
    
    def get_quat_ori(self, runs: Optional[List[int]] = None) -> np.ndarray:
        """
        Returns quaternion orientation of the run.
        """
        return np.array(self._states[runs])[:, 31:35]

    def get_stiffness(
            self, 
            which: Literal["x", "y", "z", "rot_x", "rot_y", "rot_z"], 
            run: int = 1, 
            key: Literal["hybrid", "rl", "nominal"]="hybrid"
    ) -> np.ndarray:
        """
        Returns stiffness of the run. You can further specify if you want the hybrid or the rl actions.
        """
        if key == "hybrid":
            full_stiff = np.array(self._hybrid_actions[run])[:,-7:-1]
        elif key == "rl":
            full_stiff = np.array(self._rl_actions[run])[:,-7:-1]
        elif key == "nominal":
            full_stiff = np.array(self._nominal_actions[run])[:,-7:-1]
        return full_stiff[:,EvalObserver.STIFF_INDEX_DICT[which]]

    def get_damping(
            self, 
            run: int = 1, 
            key: Literal["hybrid", "rl", "nominal"]="hybrid"
    ) -> np.ndarray:
        """
        Returns damping of the run. You can further specify if you want the hybrid or the rl actions.
        """
        if key=="hybrid":
            return np.array(self._hybrid_actions[run])[:,-1]
        elif key=="rl":
            return np.array(self._rl_actions[run])[:,-1]
        elif key=="nominal":
            return np.array(self._nominal_actions[run])[:,-1]
    
    def get_lambdas(self, runs: Optional[List[int]] = None) -> np.ndarray:
        """
        Return the mixing weights of the run.
        """
        if runs is None:
            runs = range(1,self.num_runs+1)

        min_len = min([len(self._forces[r]) for r in runs])    
        return np.array([np.array(self._states[r])[:min_len-1, -1] for r in runs])
    
    def get_forces(self, runs: Optional[List[int]] = None) -> np.ndarray:
        """
        Returns the forces, specified by the info-dictionary of the environment.
        """
        if runs is None:
            runs = range(1,self.num_runs+1)
        
        min_len = min([len(self._forces[r]) for r in runs])
        return np.array([np.array(self._forces[r][:min_len-1]) for r in runs])
    
    def get_state_forces(self, runs: Optional[List[int]] = None) -> np.ndarray:
        """
        Returns the forces, specified in the state of the environment (3-dim).
        """
        return np.array(self._states[runs])[:,41:44]
    
    def get_abs_vel(self, runs: Optional[List[int]] = None) -> np.ndarray:
        """
        Returns the absolute velocity of the EEF for the run.
        """
        if runs is None:
            runs = range(1,self.num_runs+1)

        min_len = min([len(self._states[r]) for r in runs])
        all_velocities = np.array([np.array(self._states[r])[:min_len-1, 35:38] for r in runs])
        return np.linalg.norm(all_velocities, axis=2)
    
    def get_pos_action(
            self, 
            run: int = 1, 
            key: Literal["hybrid", "rl", "nominal"]="hybrid"
    ) -> np.ndarray:
        """
        Returns positional action control of the run. Allows for specification of hybrid action or rl action.
        """
        if key=="hybrid":
            return np.array(self._hybrid_actions[run])[:,:3]
        elif key=="rl":
            return np.array(self._rl_actions[run])[:,:3]
        elif key=="nominal":
            return np.array(self._nominal_actions[run])[:,:3]
        
    def get_reward(
            self, 
            run: int = 1, 
            key: Literal["cross", "cross_x", "cross_yz", "force", "abs_vel", "dir"] = "force"
    ) -> np.ndarray:
        """Returns the reward from the trajectory. 
        Allows to specify which part of the reward is wanted.
        Since reward is saved cumulative, computes the rewards stepwise recursively.
        """
        attr = getattr(self, f"_reward_{key}_error")
        reward_parts = attr[run]

        # reward_parts is cumulative -> receive step values recursively
        reward_part_steps = []
        cum_subtract = 0.0
        for reward in reward_parts:
            step_reward = reward - cum_subtract
            cum_subtract = reward
            reward_part_steps.append(step_reward)
        return np.array(reward_part_steps)

    
    ########## saving and loading ##########
    
    def to_csv(self, filename) -> None:
        """Saves the object data into a CSV-file.
        (Helpful for loading the data in a notebook again.)
        """
        assert len(self._states.keys()) == 1, "Did not complete the creation of the csv saving for more than one run.."

        # get data
        for key in self._states.keys():
            steps_df = pd.DataFrame(data=self._steps[key], columns=["steps"])
            states_df = pd.DataFrame(data=self._states[key], columns=[f"states_{s}" for s in range(75)])  # hard coded dimension
            hybrid_actions_df = pd.DataFrame(data=self._hybrid_actions[key], columns=[f"hybrid_action_{s}" for s in range(13)])  # hard coded dimension
            reward_data = dict(
                reward_cross_x=self._reward_cross_x_error[key],
                reward_cross_yz=self._reward_cross_yz_error[key],
                reward_force=self._reward_force_error[key],
                reward_abs_vel=self._reward_abs_vel_error[key],
                reward_dir=self._reward_dir_error[key],
            )
            rewards_df = pd.DataFrame(data=reward_data)
            forces_df = pd.DataFrame(data=self._forces[key], columns=["forces"])

        # concatenate and write to csv
        full_df = pd.concat([steps_df, states_df, hybrid_actions_df, rewards_df, forces_df], axis=1)
        full_df.to_csv(filename)

    def from_csv(self, filename: Union[str, Path]) -> None:
        """Loads the objective data from a CSV-file."""

        data_df = pd.read_csv(filename, index_col=0)

        self._steps[1] = data_df["steps"].to_numpy()
        self._states[1] = data_df[[col for col in data_df.columns if col.startswith("states")]].to_numpy()
        self._hybrid_actions[1] = data_df[[col for col in data_df.columns if col.startswith("hybrid")]].to_numpy()
        self._reward_cross_x_error[1] = data_df["reward_cross_x"].to_numpy()
        self._reward_cross_yz_error[1] = data_df["reward_cross_yz"].to_numpy()
        self._reward_force_error[1] = data_df["reward_force"].to_numpy()
        self._reward_abs_vel_error[1] = data_df["reward_abs_vel"].to_numpy()
        self._reward_dir_error[1] = data_df["reward_dir"].to_numpy()
        self._forces[1] = data_df["forces"].to_numpy()

        print("Loaded EvalObserver successfully...")

    
