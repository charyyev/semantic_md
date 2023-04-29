from torch import nn

from trainer.base_trainer import BaseTrainer
from utils.eval_metrics import depth_metrics


class TripleLossTrainer(BaseTrainer):
    def __init__(self, config):
        config["data_flags"]["return_types"]["border"] = True
        super().__init__(config)

    def build_model(self):
        super().build_model()
        self.loss_depth = nn.L1Loss(reduction="none")
        self.loss_semantic = nn.CrossEntropyLoss(reduction="none", ignore_index=-2)
        self.loss_contours = nn.CrossEntropyLoss(reduction="none", ignore_index=-1)

    def step(self, data):
        input_image = data["input_image"].to(self.config["device"])
        depth = data["depths"].to(self.config["device"])
        semantic = data["input_segs"].to(self.config["device"])
        semantic = semantic.squeeze().long() - 1
        contours = data["border"].to(self.config["device"])
        contours = contours.squeeze().long()

        self.optimizer.zero_grad()

        pred_depth, pred_semantic, pred_contours = self.model(input_image)
        # clamp values to >0
        loss_depth = self.loss_depth(pred_depth, depth)
        loss_depth = self.nan_reduction(loss_depth)
        loss_semantic = self.loss_semantic(pred_semantic, semantic)
        loss_semantic = self.nan_reduction(loss_semantic)
        loss_contours = self.loss_contours(pred_contours, contours)
        loss_contours = self.nan_reduction(loss_contours)

        lam_semantic = self.config["hyperparameters"]["train"]["lambda_semantic"]
        lam_contours = self.config["hyperparameters"]["train"]["lambda_contours"]
        loss = loss_depth + lam_semantic * loss_semantic + lam_contours * loss_contours

        metrics = depth_metrics(pred_depth, depth, self.epsilon, self.config)

        full_metrics = {
            "loss": loss.item(),
            "loss_depth": loss_depth.item(),
            "loss_semantic": loss_semantic.item(),
            "loss_contours": loss_contours.item(),
            **metrics,
        }

        return loss, full_metrics

    def flag_sanity_check(self, flags):
        if flags["type"] is not None:
            raise ValueError("Cannot set flag while running multi-loss")
