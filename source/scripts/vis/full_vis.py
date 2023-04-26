import os
import sys

import numpy as np
import torch

import matplotlib as mlp
import matplotlib.pyplot as plt
from matplotlib import cm

from source.datasets import hypersim_dataset
from source.models import ModelFactory
from source.utils.configs import Config

# sys.path.insert(0, "/media/ankitaghosh/Data/ETH/3DVision/semantic_md")


class Vis:
    def __init__(self, dataset, model, config):
        self.index = 0
        self.dataset = dataset
        self.model = model
        self.config = config
        nrows, ncols = 2, 4

        figsize = 1920 / 100, 1080 / 100
        # figsize = 1200 / 100, 800 / 100
        self.fig, self.axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for row in range(nrows):
            for col in range(ncols):
                self.axes[row][col].axis("off")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        # # colorbar
        # min_depth, max_depth = self.config["transformations"]["depth_range"]
        # norm = plt.Normalize(vmin=min_depth, vmax=max_depth)
        # sm = cm.ScalarMappable(cmap='viridis', norm=norm)
        # sm.set_array([])
        # cb = self.fig.colorbar(sm, ax=self.axes[:, 2], fraction=0.9, aspect=2, ticks=[0, max_depth / 2, max_depth])
        # cb.ax.tick_params(labelsize=25)

        self.update_image()

    def _remove_texts(self):
        for ax in list(self.axes.flatten()):
            for text in ax.texts:
                text.remove()

    def update_image(self):
        self._remove_texts()
        data = self.dataset[self.index]

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
        # img_segs = (np.clip(segs, 0, config["data_flags"]["seg_classes"]) /
        # config["data_flags"]["seg_classes"] * 255).astype(int)
        img_segs = (
            np.clip(segs, 0, self.config["data_flags"]["parameters"]["seg_classes"])
        ).astype(int)
        img_segs_vir = cm.tab20b(img_segs)
        self.axes[0, 2].set_title("semantic map")
        self.axes[0, 2].imshow(img_segs_vir)

        # seg post-processing
        segs_post = data["segs"].squeeze().numpy()
        if self.config["data_flags"]["type"] == "border":
            img_segs_post = cm.tab20b(segs_post)[:, :, :3]
            self.axes[0, 3].set_title("border")
            self.axes[0, 3].imshow(img_segs_post)
        elif self.config["data_flags"]["type"] == "simplified_onehot":
            segs_post = (np.argmax(segs_post, axis=0).astype(int) / 3 * 255).astype(int)
            img_segs_post = cm.viridis(segs_post)
            self.axes[0, 3].set_title("3-encoded")
            self.axes[0, 3].imshow(img_segs_post)

        # Prediction
        input_ = data["image"].unsqueeze(0)
        if self.config["data_flags"]["type"] == "semantic_convolution":
            input_sem = data["segs"].unsqueeze(0)
            pred = self.model(input_, input_sem)
        else:
            pred = self.model(input_)
        pred = pred.detach().numpy().squeeze()
        self.axes[1, 0].set_title("prediction")
        self.axes[1, 0].imshow(pred, cmap="viridis")

        # Diff prediction ground truth
        diff = np.abs(depths - pred)
        self.axes[1, 1].set_title("difference")
        self.axes[1, 1].imshow(diff, cmap="viridis")

        self.fig.canvas.title = f"Image {self.index}"
        self.fig.canvas.draw()

    def _save_image(self):
        base_path = self.config["visualize"]["save_path"]
        os.makedirs(base_path, exist_ok=True)
        file_name = os.path.join(base_path, f"Image_{self.index}")
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


def main():
    mlp.use("TkAgg")
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

    model_path = config["visualize"]["model_path"]
    if not (model_path is None or model_path.strip() == ""):
        checkpoint = torch.load(model_path, map_location=torch.device(config["device"]))
        model.load_state_dict(checkpoint)
    model.eval()

    Vis(dataset, model, config)
    plt.show()


if __name__ == "__main__":
    main()
