import argparse

from trainer.base_trainer import BaseTrainer
from trainer.contour_trainer import ContourTrainer
from trainer.multi_loss_trainer import MultiLossTrainer
from trainer.semantic_convolution_trainer import SemanticConvolutionTrainer
from trainer.semantic_trainer import SemanticTrainer
from trainer.sobel_trainer import SobelTrainer
from trainer.triple_loss_trainer import TripleLossTrainer
from utils.configs import Config


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
    elif config["model_type"] == "semantic_baseline":
        agent = SemanticTrainer(config)
    elif config["model_type"] == "sobel_loss":
        agent = SobelTrainer(config)
    elif config["model_type"] == "multi_loss":
        agent = MultiLossTrainer(config)
    elif config["model_type"] == "triple_loss":
        agent = TripleLossTrainer(config)
    elif config["model_type"] == "contour_loss":
        agent = ContourTrainer(config)
    else:
        agent = BaseTrainer(config)
    agent.train()


if __name__ == "__main__":
    main()
