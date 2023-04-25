from utils.config import args_and_config

from trainer.base_trainer import BaseTrainer
from trainer.dual_trainer import DualTrainer

if __name__ == "__main__":
    config = args_and_config()

    if config["data_flags"]["multi-loss"]:
        agent = DualTrainer(config)
    else:
        agent = BaseTrainer(config)
    agent.train()
