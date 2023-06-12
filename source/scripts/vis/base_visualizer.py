import math
import os
import sys
from abc import abstractmethod

import numpy as np

import matplotlib.pyplot as plt
from datasets import hypersim_dataset
from matplotlib import cm
from matplotlib import image as mpimg


class BaseVisualizer:
    def __init__(self, config):
        self.config = config
        # index of the image currently being illustrated
        self.index = self.config["visualize"]["start"]
        # current downloads is used to store images for saving
        self.current_downloads = {}
        self.current_max = np.inf

        config["device"] = "cpu"
        self._init()

        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

    def _init(self):
        self._setup_data()
        self._setup_window()

    @abstractmethod
    def _setup_data(self):
        """
        Use this method to set up the dataset from which images are being visualized.
        """
        data_dir = self.config.get_subpath("data_location")
        val_file_path = self.config.get_subpath("vis_data")
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
        """
        Use this method to setup the window layout.
        """
        self.nrows, self.ncols = 2, 3
        figsize = 1920 / 100, 1080 / 100
        self.fig, self.axes = plt.subplots(
            nrows=self.nrows, ncols=self.ncols, figsize=figsize
        )
        # make not-overridden subplots transparent
        for row in range(self.nrows):
            for col in range(self.ncols):
                self.axes[row][col].axis("off")

    def _remove_texts(self):
        # remove any texts used to indicate saving, etc.
        for ax in list(self.axes.flatten()):
            for text in ax.texts:
                text.remove()

    @abstractmethod
    def _draw_image(self, data):
        """
        Use this method to specify which image is being drawn where
        """
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
        self.axes[0, 2].set_title("semantic map")
        self.axes[0, 2].imshow(img_segs_vir)

        # segmentation map post-processing (can be simplified or contour)
        segs_post = data["input_segs"].squeeze().numpy()
        if self.config["data_flags"]["type"] == "contour":
            img_segs_post = cm.tab20b(segs_post)[:, :, :3]
            self.axes[1, 0].set_title("contour")
            self.axes[1, 0].imshow(img_segs_post)
        elif self.config["data_flags"]["type"] == "simplified_onehot":
            segs_post = (np.argmax(segs_post, axis=0).astype(int) / 3 * 255).astype(int)
            img_segs_post = cm.viridis(segs_post)
            self.axes[1, 0].set_title("3-encoded")
            self.axes[1, 0].imshow(img_segs_post)

        # store images which are supposed to be saved upon pressing save button
        self.current_downloads = {
            "image": (image, False),
            "depths": (np.nan_to_num(depths, copy=False, nan=0.5), True),
            "segs": (img_segs_vir, False),
            "contour": (img_segs_post, False),
        }
        self.current_max = np.max(depths)

    def update_image(self):
        """
        Redraw canvas.
        """
        self._remove_texts()
        data = self.dataset[self.index]

        self._draw_image(data)

        self.fig.canvas.title = f"Image {self.index}"
        self.fig.canvas.draw()

    def _create_colorbar(self, save_path):
        total_max_depth_range = self.config["transformations"]["depth_range"]["max"]
        max_vis_depth_image = self.config["visualize"]["max_depth"]
        if max_vis_depth_image is None:
            max_vis_depth_image = math.ceil(self.current_max * total_max_depth_range)
            max_vis_depth_image = min(max_vis_depth_image, total_max_depth_range)
        vmax = max_vis_depth_image / total_max_depth_range

        # pull ticks closer to center
        positions = [0, vmax / 2, vmax]
        offset = vmax / 2
        positions = [((position - offset) * 0.9) + offset for position in positions]
        fig, ax = plt.subplots(figsize=[2.2, 8])
        ax.axis("off")
        norm = plt.Normalize(vmin=0, vmax=vmax)
        sm = cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        cb = fig.colorbar(
            sm,
            ax=ax,
            fraction=1,
            pad=0.0,
            shrink=1,
            aspect=12,
            ticks=positions,
        )
        cb.ax.tick_params(labelsize=60)  # Set the desired font size
        cb.set_ticklabels([0, max_vis_depth_image / 2, max_vis_depth_image])
        plt.tight_layout()
        plt.savefig(save_path)
        return vmax

    def _save_image(self):
        """
        Method used for saving images.
        Images are saved to specified folder path (in config), with special description.
        """
        # build file path
        base_path = self.config.get_subpath("saved_figures")
        folder_name = os.path.basename(
            os.path.dirname(self.config["visualize"]["model_path"])
        )
        file_name = os.path.join(base_path, folder_name, f"Image_{self.index}")
        os.makedirs(file_name, exist_ok=True)
        # notify user, that saving is completed
        self.axes[0, 0].text(
            0, -60, "Saved", bbox={"facecolor": "grey", "alpha": 0.5}, fontsize=20
        )
        plt.draw()
        # create color bar
        model_name = self.config["visualize"]["model_name"]
        vmax = self._create_colorbar(
            os.path.join(file_name, f"colorbar_{model_name}.png")
        )
        # cbar.ax.figure.savefig('colorbar.png', bbox_inches='tight', pad_inches=0)
        # save each image specified in current_downloads
        for name, (arr, cmap) in self.current_downloads.items():
            if arr is None:
                continue
            save_path = os.path.join(file_name, f"{name}_{model_name}.png")
            if cmap:
                mpimg.imsave(save_path, arr, cmap="viridis", vmin=0.0, vmax=vmax)
            else:
                mpimg.imsave(save_path, arr)

    def on_key_press(self, event):
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
