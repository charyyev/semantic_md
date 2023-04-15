import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import vispy
import matplotlib as mlp
from matplotlib import cm
from vispy import app, scene, color

from datasets import hypersim_dataset
from models.model_factory import ModelFactory
from utils.config import args_and_config
from utils.transforms import compute_transforms


class Vis():
    def __init__(self, dataset, model, config):
        self.index = 0
        self.dataset = dataset
        self.model = model
        self.config = config
        nrows, ncols = 2, 3

        figsize = 1920 / 100, 1080 / 100
        self.fig, self.axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for row in range(nrows):
            for col in range(ncols):
                self.axes[row][col].axis("off")
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # colorbar
        min_depth, max_depth = self.config["transformations"]["depth_range"]
        norm = plt.Normalize(vmin=min_depth, vmax=max_depth)
        sm = cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        cb = self.fig.colorbar(sm, ax=self.axes[:, 2], fraction=0.9, aspect=2, ticks=[0, max_depth / 2, max_depth])
        cb.ax.tick_params(labelsize=25)

        self.update_image()

    def update_image(self):
        data = self.dataset[self.index]

        # Original image
        image = data["original_image"].permute((1, 2, 0)).numpy()
        self.axes[0, 0].imshow(image)

        # Depths
        depths = data["depths"].squeeze().numpy()
        self.axes[0, 1].imshow(depths, cmap="viridis")

        # Prediction
        input_ = data["image"].unsqueeze(0)
        if self.config["data_flags"]["semantic_convolution"]:
            input_sem = data["segs"].unsqueeze(0)
            pred = model(input_, input_sem)
        else:
            pred = model(input_)
        pred = pred.detach().numpy().squeeze()
        self.axes[1, 0].imshow(pred, cmap="viridis")

        # Diff prediction ground truth
        diff = np.abs(depths - pred)
        self.axes[1, 1].imshow(diff, cmap="viridis")

        self.fig.canvas.title = f"Image {self.index}"
        self.fig.canvas.draw()

    def on_key_press(self, event):
        if event.key == 'q':
            self.update_image()
        if event.key == 'right':
            if self.index < len(self.dataset) - 1:
                self.index += 1
            self.update_image()
        elif event.key == 'left':
            if self.index > 0:
                self.index -= 1
            self.update_image()
        elif event.key == 'q':
            self.destroy()

    def destroy(self):
        plt.close(self.fig)
        exit(0)


if __name__ == "__main__":
    mlp.use("TkAgg")
    config = args_and_config()

    config["device"] = "cpu"
    dataset_root_dir = config["data_location"]
    data_flags = config["data_flags"]

    pretrained_weights_path = os.path.join(config["root_dir"], "models", "pretrained_weights")
    model, transform_config = ModelFactory().get_model(config["model"], pretrained_weights_path, config,
                                                       in_channels=3)
    model.to(config["device"])
    image_transform, depth_transform, seg_transform = compute_transforms(transform_config, config)

    dataset = hypersim_dataset.HyperSimDataset(root_dir=dataset_root_dir, file_path=config["val"]["data"],
                                               image_transform=image_transform, depth_transform=depth_transform,
                                               seg_transform=seg_transform, data_flags=config["data_flags"])

    model_path = config["load"].get("path", None)
    if not (model_path is None or model_path.strip() == ""):
        checkpoint = torch.load(model_path, map_location=torch.device(config["device"]))
        model.load_state_dict(checkpoint)
    model.eval()

    vis = Vis(dataset, model, config)
    plt.show()
