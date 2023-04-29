import matplotlib as mlp
from scripts.vis.base_visualizer import BaseVisualizer
from scripts.vis.model_visualizer import ModelVisualizer
from scripts.vis.multi_visualizer import MultiVisualizer
from scripts.vis.triple_visualizer import TripleVisualizer
from utils.configs import Config


def main():
    mlp.use("TkAgg")
    config = Config()
    model_type = config["visualize"]["model_type"]
    if model_type is None:
        vis = BaseVisualizer(config)
    elif model_type == "multi_loss":
        vis = MultiVisualizer(config)
    elif model_type == "triple_loss":
        vis = TripleVisualizer(config)
    else:
        vis = ModelVisualizer(config)
    vis.run()


if __name__ == "__main__":
    main()
