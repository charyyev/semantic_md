from torch import nn

from trainer.base_trainer import BaseTrainer
from utils.eval_metrics import border_metrics, depth_metrics
from utils.loss_functions import BerHuLoss


class SobelTrainer(BaseTrainer):
    def __init__(self, config):
        config["data_flags"]["return_types"]["border"] = True
        super().__init__(config)

    def build_model(self):
        super().build_model()
        if self.config["hyperparameters"]["train"]["depth_loss_type"] == "L1":
            self.loss_depth = nn.L1Loss(reduction="none")
        elif self.config["hyperparameters"]["train"]["depth_loss_type"] == "berhu":
            self.loss_depth = BerHuLoss(contains_nan=True)

        self.loss_sobel = nn.CrossEntropyLoss(reduction="none")

    def step(self, data):
        input_image = data["input_image"].to(self.config["device"])
        depth = data["depths"].to(self.config["device"])
        sobel = data["sobel"].to(self.config["device"])
        sobel = sobel.squeeze().long()

        self.optimizer.zero_grad()

        pred_depth, pred_sobel = self.model(input_image)
        # clamp values to >0
        loss_depth = self.loss_depth(pred_depth, depth)
        loss_depth = self.nan_reduction(loss_depth)

        loss_sobel = self.loss_sobel(pred_sobel, sobel)
        loss_sobel = self.nan_reduction(loss_sobel)

        lam_sobel = self.config["hyperparameters"]["train"]["lambda_sobel"]
        loss = loss_depth + lam_sobel * loss_sobel

        metrics_depth = depth_metrics(pred_depth, depth, self.epsilon, self.config)
        metrics_sobel = border_metrics(pred_sobel, sobel, self.epsilon, self.config)

        full_metrics = {
            "loss": loss.item(),
            "loss_depth": loss_depth.item(),
            "loss_sobel": loss_sobel.item(),
            **metrics_depth,
            **metrics_sobel,
        }

        return loss, full_metrics

    def flag_sanity_check(self, flags):
        if flags["type"] is not None:
            raise ValueError("Cannot set flag while running multi-loss")
