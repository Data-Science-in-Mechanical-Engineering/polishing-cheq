from omegaconf import DictConfig
from hydra.utils import instantiate
import hydra
import sys
import wandb
from async_learner.async_actor import AsyncActor
from utils.launcher_utils import instantiate_environment
from environments.wrappers.record_episode_statistics import RecordEpisodeStatistics


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig):

    # instantiate environment
    base_env = instantiate_environment(cfg=cfg)
    env = RecordEpisodeStatistics(env=base_env)

    # instantiate rl agent
    rl_agent_actor = instantiate(cfg.rl, env=env)

    # instantiate nominal agent
    nominal_agent = instantiate(cfg.nominal, env=env)

    # instantiate hybrid agent
    hybrid_agent = instantiate(cfg.hybrid, rl_agent=rl_agent_actor, nominal_agent=nominal_agent)

    # instantiate actor
    actor = AsyncActor(
        agent=hybrid_agent, 
        env=env, 
        actor_config=cfg.training.async_actor,
        name_run=cfg.dir
    )

    # start actor loop
    actor.act()

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
