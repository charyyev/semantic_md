import os
import sys
from abc import abstractmethod

import numpy as np

import matplotlib.pyplot as plt
from datasets import hypersim_dataset
from matplotlib import cm


class BaseVisualizer:
    def __init__(self, config):
        self.index = 0
        self.config = config

        config["device"] = "cpu"
        self._init()

        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

    def _init(self):
        self._setup_data()
        self._setup_window()

    @abstractmethod
    def _setup_data(self):
        data_dir = self.config.get_subpath("data_location")
        val_file_path = self.config.get_subpath("val_data")
        self.dataset = hypersim_dataset.HyperSimDataset(
            data_dir=data_dir,
            file_path=val_file_path,
            image_transform=None,
            depth_transform=None,
            seg_transform=None,
            data_flags=self.config["data_flags"],
        )

    @abstractmethod
    def _setup_window(self):
        self.nrows, self.ncols = 2, 2
        figsize = 1920 / 100, 1080 / 100
        self.fig, self.axes = plt.subplots(
            nrows=self.nrows, ncols=self.ncols, figsize=figsize
        )
        for row in range(self.nrows):
            for col in range(self.ncols):
                self.axes[row][col].axis("off")

    def _remove_texts(self):
        for ax in list(self.axes.flatten()):
            for text in ax.texts:
                text.remove()

    @abstractmethod
    def _draw_image(self, data):
        # Original image
        image = data["original_image"].permute((1, 2, 0)).numpy()
        self.axes[0, 0].set_title("image")
        self.axes[0, 0].imshow(image)

        # Depths
        depths = data["depths"].squeeze().numpy()
        self.axes[0, 1].set_title("depth map")
        self.axes[0, 1].imshow(depths, cmap="viridis")

        # segmentation
        segs = data["original_seg"].squeeze().numpy()
        img_segs = (
            np.clip(segs, 0, self.config["data_flags"]["parameters"]["seg_classes"])
        ).astype(int)
        img_segs_vir = cm.tab20b(img_segs)
        self.axes[1, 0].set_title("semantic map")
        self.axes[1, 0].imshow(img_segs_vir)

        # seg post-processing
        segs_post = data["input_segs"].squeeze().numpy()
        if self.config["data_flags"]["type"] == "border":
            img_segs_post = cm.tab20b(segs_post)[:, :, :3]
            self.axes[1, 1].set_title("border")
            self.axes[1, 1].imshow(img_segs_post)
        elif self.config["data_flags"]["type"] == "simplified_onehot":
            segs_post = (np.argmax(segs_post, axis=0).astype(int) / 3 * 255).astype(int)
            img_segs_post = cm.viridis(segs_post)
            self.axes[1, 1].set_title("3-encoded")
            self.axes[1, 1].imshow(img_segs_post)

    def update_image(self):
        self._remove_texts()
        data = self.dataset[self.index]

        self._draw_image(data)

        self.fig.canvas.title = f"Image {self.index}"
        self.fig.canvas.draw()

    def _save_image(self):
        base_path = self.config.get_subpath("saved_figures")
        os.makedirs(base_path, exist_ok=True)
        file_name = os.path.join(
            base_path, f"Image_{self.index}_{self.config['visualize']['model_path']}"
        )
        plt.savefig(file_name)
        self.axes[0, 0].text(
            0, -60, "Saved", bbox={"facecolor": "grey", "alpha": 0.5}, fontsize=20
        )
        plt.draw()

    def on_key_press(self, event):
        print(event.key)
        if event.key in {"right", "d"}:
            if self.index < len(self.dataset) - 1:
                self.index += 1
            self.update_image()
        elif event.key in {"left", "a"}:
            if self.index > 0:
                self.index -= 1
            self.update_image()
        elif event.key == "q":
            self.destroy()
        elif event.key == " ":  # space bar
            self._save_image()
        else:
            UserWarning(f"Key is not mapped to function: {event.key}")

    def destroy(self):
        plt.close(self.fig)
        sys.exit(0)

    def run(self):
        self.update_image()
        plt.show()
