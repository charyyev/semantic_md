from torch import nn

from trainer.base_trainer import BaseTrainer
from utils.eval_metrics import border_metrics, depth_metrics, seg_metrics
from utils.loss_functions import BerHuLoss, DiceLoss, FocalTverskyLoss


class TripleLossTrainer(BaseTrainer):
    def __init__(self, config):
        config["data_flags"]["return_types"]["border"] = True
        super().__init__(config)

    def build_model(self):
        super().build_model()
        if self.config["hyperparameters"]["train"]["depth_loss_type"] == "L1":
            self.loss_depth = nn.L1Loss(reduction="none")
        elif self.config["hyperparameters"]["train"]["depth_loss_type"] == "berhu":
            self.loss_depth = BerHuLoss(contains_nan=True)

        if self.config["hyperparameters"]["train"]["semantic_loss_type"] == "CE":
            self.loss_semantic = nn.CrossEntropyLoss(reduction="none", ignore_index=-1)
        elif self.config["hyperparameters"]["train"]["semantic_loss_type"] == "Dice":
            self.loss_semantic = DiceLoss(num_encode=self.config["data_flags"]["parameters"]["seg_classes"])
        elif self.config["hyperparameters"]["train"]["semantic_loss_type"] == "FTL":
            self.loss_semantic = FocalTverskyLoss(
                alpha=self.config["hyperparameters"]["train"]["weight_alpha"],
                gamma=self.config["hyperparameters"]["train"]["weight_gamma"],
                num_encode=self.config["data_flags"]["parameters"]["seg_classes"]
                )
        elif self.config["hyperparameters"]["train"]["semantic_loss_type"] == "Dice_CE":
            self.loss_semanticCE = nn.CrossEntropyLoss(reduction="none", ignore_index=-1)
            self.loss_semanticOther = DiceLoss(num_encode=self.config["data_flags"]["parameters"]["seg_classes"])
        elif self.config["hyperparameters"]["train"]["semantic_loss_type"] == "FTL_CE":
            self.loss_semanticCE = nn.CrossEntropyLoss(reduction="none", ignore_index=-1)
            self.loss_semanticOther = FocalTverskyLoss(
                alpha=self.config["hyperparameters"]["train"]["weight_alpha"],
                gamma=self.config["hyperparameters"]["train"]["weight_gamma"],
                num_encode=self.config["data_flags"]["parameters"]["seg_classes"]
            )

        self.loss_contours = nn.CrossEntropyLoss(reduction="none")

    def step(self, data):
        input_image = data["input_image"].to(self.config["device"])
        depth = data["depths"].to(self.config["device"])
        semantic = data["original_seg"].to(self.config["device"])
        semantic = semantic.squeeze().long()
        contours = data["border"].to(self.config["device"])
        contours = contours.squeeze().long()

        self.optimizer.zero_grad()

        pred_depth, pred_semantic, pred_contours = self.model(input_image)
        # clamp values to >0
        loss_depth = self.loss_depth(pred_depth, depth)
        loss_depth = self.nan_reduction(loss_depth)

        # loss_semantic = self.loss_semantic(pred_semantic, semantic)
        # loss_semantic = self.nan_reduction(loss_semantic)
        if (self.config["hyperparameters"]["train"]["semantic_loss_type"] == "Dice_CE"
            ) or (
            self.config["hyperparameters"]["train"]["semantic_loss_type"] == "FTL_CE"
            ):
            loss_semanticCE = self.loss_semanticCE(pred_semantic, semantic)
            loss_semanticCE = self.nan_reduction(loss_semanticCE)
            loss_semanticOther = self.loss_semanticOther(pred_semantic, semantic)
            loss_semantic = (loss_semanticCE * 
                             self.config["hyperparameters"]["train"]["weight_lambda"]
                             ) + (loss_semanticOther * 
                                  (1- self.config["hyperparameters"]["train"]["weight_lambda"])
                                  )
        else:
            loss_semantic = self.loss_semantic(pred_semantic, semantic)
        
        loss_contours = self.loss_contours(pred_contours, contours)
        loss_contours = self.nan_reduction(loss_contours)

        lam_semantic = self.config["hyperparameters"]["train"]["lambda_semantic"]
        lam_contours = self.config["hyperparameters"]["train"]["lambda_contours"]
        loss = loss_depth + lam_semantic * loss_semantic + lam_contours * loss_contours

        metrics_depth = depth_metrics(pred_depth, depth, self.epsilon, self.config)
        metrics_seg = seg_metrics(pred_semantic, semantic, self.epsilon, self.config)
        metrics_contour = border_metrics(
            pred_contours, contours, self.epsilon, self.config
        )

        full_metrics = {
            "loss": loss.item(),
            "loss_depth": loss_depth.item(),
            "loss_semantic": loss_semantic.item(),
            "loss_contours": loss_contours.item(),
            **metrics_depth,
            **metrics_seg,
            **metrics_contour,
        }

        return loss, full_metrics

    def flag_sanity_check(self, flags):
        if flags["type"] is not None:
            raise ValueError("Cannot set flag while running multi-loss")
