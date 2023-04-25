from source.trainer.base_trainer import BaseTrainer
from source.trainer.semantic_convolution_trainer import SemanticConvolutionTrainer
from source.utils.configs import Config


def main():
    config = Config()

    if config["data_flags"]["type"] == "semantic_convolution":
        agent = SemanticConvolutionTrainer(config)
    else:
        agent = BaseTrainer(config)
    agent.train()


if __name__ == "__main__":
    main()
