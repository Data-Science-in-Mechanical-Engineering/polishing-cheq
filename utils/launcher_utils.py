import gymnasium
from agentlace.trainer import TrainerConfig
from utils.logging_utils import WandBLogger
import ml_collections
from omegaconf import DictConfig, OmegaConf
from typing import Union, Type, Dict, Optional, List
import gymnasium as gym

#from environments.real_robot_env import RealRobotEnv
from environments.cart_pole_env import CartPoleEnv


def make_trainer_config(
        port_number: int = 5488, 
        broadcast_port: int = 5489, 
        request_types: List[str] = None
    ):
    """
    Creates the trainer configuration for the server and the client.

    Args:
        port_number (int, optional): The port number. Defaults to 5488.
        broadcast_port (int, optional): The broadcast port. Defaults to 5489.
        request_types (List[str], optional): The possible request types.
            If not specified, defaults to ['send-stats', 'send-eval-metrics', 'pause-training', 'continue-training', 'store-best-model'].

    Returns:
        TrainerConfig: The trainer configurations for the Server and the Client.
    """
    if request_types:
        r_types = request_types
    else:
        r_types = ["send-stats", "send-eval-metrics", "pause-training", "continue-training", "store-best-model"]

    return TrainerConfig(
        port_number=port_number,
        broadcast_port=broadcast_port,
        request_types=r_types,
    )


def make_wandb_logger(project: str = "",
                      group: str = "testing",
                      job_type: str = "training",
                      debug: bool = False,
                      entity: str = None):
    
    wandb_config = ml_collections.ConfigDict()
    wandb_config.update({"project": project,
                         "group": group,
                         "job_type": job_type,
                         "entity": entity})
    
    wandb_logger = WandBLogger(wandb_config=wandb_config, debug=debug)

    return wandb_logger


def convert_to_dict(cfg: Union[DictConfig, dict]):
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    return cfg


def instantiate_environment(cfg: DictConfig) -> Type[gym.Env]:
    """
    Initialize the environment based on the config specifications.

    Args:
        cfg (DictConfig): The config file

    Returns:
        gym.env instance
    """
    env_type = cfg.get("environment", {}).get("name")
    env_lst = ["sim_robot", "cart_pole"]
    assert env_type in env_lst, f"Environment must be one of {env_lst}!"

    if env_type == "real_robot":
        # env = RealRobotEnv(env_config=cfg.environment, task_config=cfg.tasks)
        raise
    elif env_type == "sim_robot":
        from environments.sim_robot_env import SimRobotEnv
        env = SimRobotEnv(env_config=cfg.environment, task_config=cfg.tasks)
    elif env_type == "cart_pole":
        env = CartPoleEnv(env_config=cfg.environment)
    
    return env
