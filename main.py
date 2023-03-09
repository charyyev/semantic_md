import argparse
import json

from trainer import Trainer

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--config', type=str, nargs='?', default = "/home/sapar/3dvision/code/configs/base.json")
    # Parse the argument
    args = parser.parse_args()


    f = open(args.config)
    config = json.load(f)

    agent = Trainer(config)
    agent.train()
