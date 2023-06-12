import numpy as np
import torch

from datasets import hypersim_dataset
from matplotlib import cm
from matplotlib import pyplot as plt
from models import ModelFactory
from scripts.vis.base_visualizer import BaseVisualizer


class MultiVisualizer(BaseVisualizer):
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
            self.model.load_state_dict(checkpoint, strict=self.config["load"]["strict"])
        self.model.eval()

    def _setup_window(self):
        self.nrows, self.ncols = 2, 3
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
            depths,
            cmap="viridis",
        )

        # segmentation
        segs = data["original_seg"].squeeze().numpy()
        img_segs = (
            np.clip(segs, 0, self.config["data_flags"]["parameters"]["seg_classes"])
        ).astype(int)
        img_segs_vir = cm.tab20b(img_segs)
        self.axes[0, 2].set_title("semantic map")
        self.axes[0, 2].imshow(img_segs_vir)

        # Prediction
        input_ = data["input_image"].unsqueeze(0)
        pred_depth, pred_semantic = self.model(input_)
        pred_depth = pred_depth.detach().numpy().squeeze()
        self.axes[1, 1].set_title("prediction depth")
        self.axes[1, 1].imshow(
            pred_depth,
            cmap="viridis",
        )

        pred_semantic = torch.argmax(pred_semantic, dim=1)
        pred_semantic = pred_semantic.detach().numpy().squeeze()
        pred_semantic = cm.tab20b(pred_semantic)
        self.axes[1, 2].set_title("prediction semantic")
        self.axes[1, 2].imshow(pred_semantic)

        # Diff prediction ground truth
        diff = np.abs(depths - pred_depth)
        self.axes[1, 0].set_title("difference")
        self.axes[1, 0].imshow(
            diff,
            cmap="viridis",
        )

        # square image
        square_length = image.shape[0]
        diff = (image.shape[1] - square_length) // 2
        save_image = image[:, diff:-diff]
        self.current_downloads = {
            "image": (save_image, False),
            "depths": (np.nan_to_num(depths, copy=False, nan=0.5), True),
            "pred_depth": (pred_depth, True),
            "pred_semantic": (pred_semantic, True),
            "segs": (img_segs_vir, False),
        }
        self.current_max = max(np.max(depths), np.max(pred_depth))
