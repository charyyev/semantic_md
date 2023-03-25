from utils.config import args_and_config

from trainer import Trainer

if __name__ == "__main__":
    config = args_and_config()

    agent = Trainer(config)
    agent.train()
