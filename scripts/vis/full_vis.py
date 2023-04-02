import os

import numpy as np
import torch

import vispy
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from vispy.scene import SceneCanvas
from vispy import app

from datasets import hypersim_dataset
from datasets.nyu_dataset import NyuDataset
from models.model_factory import ModelFactory
from utils.config import args_and_config
from matplotlib import cm
import matplotlib.pyplot as plt

from utils.transforms import compute_transforms


class Vis():
    def __init__(self, dataset, model):
        self.index = 0
        self.dataset = dataset
        self.model = model

        self.canvas = SceneCanvas(keys='interactive',
                                  show=True,
                                  size=(1280, 1280))
        self.canvas.events.key_press.connect(self._key_press)
        self.canvas.events.draw.connect(self._draw)
        self.canvas.show()

        self.view = self.canvas.central_widget.add_view()
        self.image = vispy.scene.visuals.Image(parent=self.view.scene)

        self.update_image()

    def update_image(self):
        grid = (2, 3)  # nrows, ncols
        fig, axes = plt.subplots(nrows=grid[0], ncols=grid[1], figsize=(25, 12))
        for row in range(grid[0]):
            for col in range(grid[1]):
                axes[row, col].axis('off')

        # image
        data = self.dataset[self.index]
        image = data["image"].permute((1, 2, 0))
        image = image.numpy()
        axes[0, 0].imshow(image)

        # depths
        depths = data["depths"].squeeze().numpy()
        img_depths = depths
        img_depths_vir = cm.viridis(img_depths)[:, :, :3]
        axes[0, 1].imshow(img_depths_vir)

        # colorbar
        norm = plt.Normalize(vmin=0, vmax=1)
        sm = cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=axes[0, 2], fraction=0.9, pad=0.04, shrink=.9, aspect=1.5, ticks=[0, 1])
        cb.ax.tick_params(labelsize=25)

        # prediction
        input_ = data["data"].unsqueeze(0)
        pred = model(input_)
        pred = pred.detach().numpy().squeeze()
        axes[1, 0].imshow(pred)

        # diff prediction ground truth
        diff = np.abs(img_depths - pred)  # TODO: abs or no
        diff_vir = cm.viridis(diff)[:, :, :3]
        axes[1, 1].imshow(diff_vir)

        # get the figure as a numpy ndarray
        plt.show()
        renderer = fig.canvas.get_renderer()
        img = np.frombuffer(renderer.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        self.canvas.title = str(self.index)
        self.image.set_data(img)
        self.view.camera = vispy.scene.PanZoomCamera(aspect=1)
        self.view.camera.set_range()
        self.view.camera.flip = (0, 1, 0)
        self.canvas.update()

    def _key_press(self, event):
        if event.key == 'Right':
            if self.index < len(self.dataset) - 1:
                self.index += 1
            self.update_image()

        if event.key == 'Left':
            if self.index > 0:
                self.index -= 1
            self.update_image()

        if event.key == 'Q':
            self.destroy()

    def destroy(self):
        # destroy the visualization
        self.canvas.close()
        vispy.app.quit()

    def _draw(self, event):
        if self.canvas.events.key_press.blocked():
            self.canvas.events.key_press.unblock()

    def run(self):
        self.canvas.app.run()


if __name__ == "__main__":
    config = args_and_config()

    dataset_root_dir = config["data_location"]
    data_flags = config["data_flags"]

    in_channels = 3
    if config["data_flags"]["concat"]:
        if config["data_flags"]["onehot"]:
            in_channels = 3 + config["data_flags"]["seg_classes"]
        else:
            in_channels = 4

    pretrained_weights_path = os.path.join(config["root_dir"], "models", "pretrained_weights")
    model, transform_config = ModelFactory().get_model(config["model"], pretrained_weights_path,
                                                       in_channels=in_channels, classes=1)
    model.to(config["device"])
    image_transform, depth_transform, seg_transform = compute_transform = compute_transforms(hypersim_dataset,
                                                                                             transform_config, config)

    dataset = hypersim_dataset.HyperSimDataset(root_dir=dataset_root_dir, train=False,
                                               image_transform=image_transform,
                                               depth_transform=depth_transform, seg_transform=seg_transform,
                                               data_flags=config["data_flags"])

    model_path = config["load"].get("path", None)
    if not (model_path is None or model_path.strip() == ""):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
    model.eval()

    vis = Vis(dataset, model)
    vis.run()
