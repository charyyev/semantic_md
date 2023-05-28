import numpy as np
import torch
from torch import nn

from trainer.base_trainer import BaseTrainer
from utils.conversions import depth_to_sobel
from utils.eval_metrics import border_metrics, depth_metrics
from utils.loss_functions import BerHuLoss


class ContourTrainer(BaseTrainer):
    def __init__(self, config):
        config["data_flags"]["return_types"]["border"] = True
        super().__init__(config)

    def build_model(self):
        super().build_model()
        if self.config["hyperparameters"]["train"]["depth_loss_type"] == "L1":
            self.loss_depth = nn.L1Loss(reduction="none")
        elif self.config["hyperparameters"]["train"]["depth_loss_type"] == "berhu":
            self.loss_depth = BerHuLoss(contains_nan=True)

        self.loss_contours = nn.CrossEntropyLoss(reduction="none")

        self.loss_sobel = nn.BCELoss(reduction="none")

    def step(self, data):
        input_image = data["input_image"].to(self.config["device"])
        depth = data["depths"].to(self.config["device"])
        contours = data["border"].to(self.config["device"])
        contours = contours.squeeze(1).long()

        self.optimizer.zero_grad()

        pred_depth, pred_contours = self.model(input_image)

        pred_depth_to_numpy = pred_depth.squeeze(1).clone().detach().cpu().numpy()
        pred_sobel = np.zeros_like(pred_depth_to_numpy)
        for i in range(pred_depth_to_numpy.shape[0]):
            pred_sobel[i] = depth_to_sobel(
                pred_depth_to_numpy[i],
                self.config["data_flags"]["parameters"]["sobel_ksize"],
                self.config["data_flags"]["parameters"]["sobel_threshold"],
            )
        pred_sobel = torch.tensor(pred_sobel, requires_grad=True).float().cuda()
        pred_sobel[pred_sobel == 0] = self.epsilon
        pred_sobel[pred_sobel == 1] = 1 - self.epsilon

        # clamp values to >0
        loss_depth = self.loss_depth(pred_depth, depth)
        loss_depth = self.nan_reduction(loss_depth)

        loss_contours = self.loss_contours(pred_contours, contours)
        loss_contours = self.nan_reduction(loss_contours)

        loss_sobel = self.loss_sobel(pred_sobel, contours.float())
        loss_sobel = self.nan_reduction(loss_sobel)

        lam_sobel = self.config["hyperparameters"]["train"]["lambda_sobel"]
        lam_contours = self.config["hyperparameters"]["train"]["lambda_contours"]
        loss = loss_depth + lam_contours * loss_contours + lam_sobel * loss_sobel

        metrics_depth = depth_metrics(pred_depth, depth, self.epsilon, self.config)
        metrics_sobel = border_metrics(
            pred_contours, contours, self.epsilon, self.config
        )

        full_metrics = {
            "loss": loss.item(),
            "loss_depth": loss_depth.item(),
            "loss_contour": loss_contours.item(),
            "loss_sobel": loss_sobel.item(),
            **metrics_depth,
            **metrics_sobel,
        }

        return loss, full_metrics

    def flag_sanity_check(self, flags):
        if flags["type"] is not None:
            raise ValueError("Cannot set flag while running multi-loss")
