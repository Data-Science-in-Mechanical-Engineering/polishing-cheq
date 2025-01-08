from omegaconf import DictConfig
from hydra.utils import instantiate
import hydra
import sys
import wandb
import gymnasium as gym
from utils.logging_utils import WandBLogger
from utils.launcher_utils import instantiate_environment
from async_learner.async_trainer import AsyncTrainer
from environments.wrappers.record_episode_statistics import RecordEpisodeStatistics


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig):

    # instantiate wandb logger and log config
    wandb_logger = WandBLogger(wandb_config=cfg.wandb)
    wandb_logger.prepare_logger(cfg)

    # instantiate environment
    base_env = instantiate_environment(cfg)
    env = RecordEpisodeStatistics(base_env)

    # instantiate rl agent
    rl_agent_learner = instantiate(cfg.rl, env=env)

    # instantiate actor and learner
    learner = AsyncTrainer(agent=rl_agent_learner, env=env, wandb_logger=wandb_logger, learner_config=cfg.training.async_learner)

    # distinguish between load pretrained or start with new one
    if cfg.stage == "load":
        learner.load_checkpoint(cfg.model_load_path)
    elif cfg.stage == "train":
        pass
    else:
        raise Exception(f"Only 'load' and 'train' are enabled as a stage with this script.")

    # start learner loop
    learner.learn()

if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        # handle keyboard interruption smoothly
        print("Keyboard Interruption was triggered.")
        wandb.finish()
        sys.exit(0)

    except Exception:
        raise
