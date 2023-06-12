import numpy as np
import torch

from datasets import hypersim_dataset
from matplotlib import cm
from matplotlib import pyplot as plt
from models import ModelFactory
from scripts.vis.base_visualizer import BaseVisualizer


class ModelVisualizer(BaseVisualizer):
    def _setup_data(self):
        self.config["model_type"] = self.config["visualize"]["model_type"]
        self.model, transform_config = ModelFactory().get_model(
            self.config, in_channels=3
        )
        self.model.to(self.config["device"])

        (
            image_transform,
            depth_transform,
            seg_transform,
        ) = hypersim_dataset.compute_transforms(transform_config, self.config)

        data_dir = self.config.get_subpath("data_location")
        val_file_path = self.config.get_subpath("val_data")

        self.dataset = hypersim_dataset.HyperSimDataset(
            data_dir=data_dir,
            file_path=val_file_path,
            image_transform=image_transform,
            depth_transform=depth_transform,
            seg_transform=seg_transform,
            data_flags=self.config["data_flags"],
        )

        model_path = self.config["visualize"]["model_path"]
        if not (model_path is None or model_path.strip() == ""):
            checkpoint = torch.load(
                model_path, map_location=torch.device(self.config["device"])
            )
            self.model.load_state_dict(checkpoint)
        self.model.eval()

    def _setup_window(self):
        self.nrows, self.ncols = 2, 4
        figsize = 1920 / 100, 1080 / 100
        self.fig, self.axes = plt.subplots(
            nrows=self.nrows, ncols=self.ncols, figsize=figsize
        )
        for row in range(self.nrows):
            for col in range(self.ncols):
                self.axes[row][col].axis("off")

    def _draw_image(self, data):
        # Original image
        image = data["original_image"].permute((1, 2, 0)).numpy()
        self.axes[0, 0].set_title("image")
        self.axes[0, 0].imshow(image)

        # Depths
        depths = data["depths"].squeeze().numpy()
        self.axes[0, 1].set_title("depth map")
        self.axes[0, 1].imshow(
            depths, cmap="viridis", vmin=self.min_depth, vmax=self.max_depth
        )

        # segmentation
        segs = data["original_seg"].squeeze().numpy()
        img_segs = (
            np.clip(segs, 0, self.config["data_flags"]["parameters"]["seg_classes"])
        ).astype(int)
        img_segs_vir = cm.tab20b(img_segs)
        self.axes[0, 2].set_title("semantic map")
        self.axes[0, 2].imshow(img_segs_vir)

        # seg post-processing
        segs_post = data["input_segs"].squeeze().numpy()
        img_segs_post = None
        if self.config["data_flags"]["type"] == "contour":
            img_segs_post = cm.tab20b(segs_post)[:, :, :3]
            self.axes[0, 3].set_title("contour")
            self.axes[0, 3].imshow(img_segs_post)
        elif self.config["data_flags"]["type"] == "simplified_onehot":
            segs_post = (np.argmax(segs_post, axis=0).astype(int) / 3 * 255).astype(int)
            img_segs_post = cm.viridis(segs_post)
            self.axes[0, 3].set_title("3-encoded")
            self.axes[0, 3].imshow(img_segs_post)

        # Prediction
        input_ = data["input_image"].unsqueeze(0)
        if self.config["data_flags"]["type"] == "semantic_convolution":
            input_sem = data["input_segs"].unsqueeze(0)
            pred = self.model(input_, input_sem)
        else:
            pred = self.model(input_)
        pred = pred.detach().numpy().squeeze()
        self.axes[1, 0].set_title("prediction")
        self.axes[1, 0].imshow(
            pred, cmap="viridis", vmin=self.min_depth, vmax=self.max_depth
        )

        # Diff prediction ground truth
        diff = np.abs(depths - pred)
        self.axes[1, 1].set_title("difference")
        self.axes[1, 1].imshow(
            diff, cmap="viridis", vmin=self.min_depth, vmax=self.max_depth
        )

        # square image
        square_length = image.shape[0]
        diff = (image.shape[1] - square_length) // 2
        save_image = image[:, diff:-diff]
        self.current_downloads = {
            "image": (save_image, False),
            "depths": (np.nan_to_num(depths, copy=False, nan=0.5), True),
            "pred": (pred, True),
            "segs": (img_segs_vir, False),
            "img_segs_post": (img_segs_post, False),
        }
        self.current_max = max(np.max(depths), np.max(pred))
