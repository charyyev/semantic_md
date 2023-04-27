import argparse

from source.trainer.base_trainer import BaseTrainer
from source.trainer.multi_loss_trainer import MultiLossTrainer
from source.trainer.semantic_convolution_trainer import SemanticConvolutionTrainer
from source.utils.configs import Config


def main():
    parser = argparse.ArgumentParser(description="Argparser")
    parser.add_argument(
        "-c",
        "--config",
        help="specify config name",
        type=str,
        action="store",
        default="user",
    )
    args = parser.parse_args()
    config = Config(file=args.config)

    if config["data_flags"]["type"] == "semantic_convolution":
        agent = SemanticConvolutionTrainer(config)
    elif config["model_type"] == "multi_loss":
        agent = MultiLossTrainer(config)
    else:
        agent = BaseTrainer(config)
    agent.train()


if __name__ == "__main__":
    main()
