import os

import numpy as np

import matplotlib.pyplot as plt
import vispy
from matplotlib import cm
from vispy import app
from vispy.scene import SceneCanvas

from source.datasets import hypersim_dataset
from source.models import ModelFactory
from source.utils.configs import Config


class Vis:
    def __init__(self, dataset):
        self.index = 0
        self.dataset = dataset

        self.canvas = SceneCanvas(keys="interactive", show=True, size=(1280, 1280))
        self.canvas.events.key_press.connect(self._key_press)
        self.canvas.events.draw.connect(self._draw)
        self.canvas.show()

        self.view = self.canvas.central_widget.add_view()
        self.image = vispy.scene.visuals.Image(parent=self.view.scene)

        self.update_image()

    def update_image(self):
        data = self.dataset[self.index]
        image = data["image"].permute((1, 2, 0))
        depths = data["depths"].squeeze().numpy()

        image = image.numpy()
        img_depths = (depths - np.min(depths)) / np.max(depths)
        img_depths = cm.viridis(img_depths)[:, :, :3]
        norm = plt.Normalize(vmin=0, vmax=1)
        sm = cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])

        grid = (2, 3)  # nrows, ncols
        fig, axes = plt.subplots(nrows=grid[0], ncols=grid[1], figsize=(25, 12))
        for row in range(grid[0]):
            for col in range(grid[1]):
                axes[row, col].axis("off")
        axes[0, 0].imshow(image)
        axes[0, 1].imshow(img_depths)
        cb = fig.colorbar(
            sm,
            ax=axes[0, 2],
            fraction=0.9,
            pad=0.04,
            shrink=0.9,
            aspect=1.5,
            ticks=[0, 1],
        )
        cb.ax.tick_params(labelsize=25)

        plt.show()
        renderer = fig.canvas.get_renderer()

        # get the figure as a numpy ndarray
        img = np.frombuffer(renderer.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        self.canvas.title = str(self.index)
        self.image.set_data(img)
        self.view.camera = vispy.scene.PanZoomCamera(aspect=1)
        self.view.camera.set_range()
        self.view.camera.flip = (0, 1, 0)
        self.canvas.update()

    def _key_press(self, event):
        if event.key == "Right":
            if self.index < len(self.dataset) - 1:
                self.index += 1
            self.update_image()

        if event.key == "Left":
            if self.index > 0:
                self.index -= 1
            self.update_image()

        if event.key == "Q":
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


def main():
    config = Config()

    config["device"] = "cpu"

    model, transform_config = ModelFactory().get_model(config, in_channels=3)
    model.to(config["device"])
    (
        image_transform,
        depth_transform,
        seg_transform,
    ) = hypersim_dataset.compute_transforms(transform_config, config)

    data_dir = config.get_subpath("data_location")
    val_file_path = config.get_subpath("val_data")

    dataset = hypersim_dataset.HyperSimDataset(
        data_dir=data_dir,
        file_path=val_file_path,
        image_transform=image_transform,
        depth_transform=depth_transform,
        seg_transform=seg_transform,
        data_flags=config["data_flags"],
    )

    vis = Vis(dataset)
    vis.run()


if __name__ == "__main__":
    main()
