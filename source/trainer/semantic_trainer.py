from torch import nn

from trainer.base_trainer import BaseTrainer
from utils.eval_metrics import seg_metrics
from utils.loss_functions import DiceLoss, FocalTverskyLoss


class SemanticTrainer(BaseTrainer):
    """
    Trainer for baseline semantic
    """

    def build_model(self):
        super().build_model()

        if self.config["hyperparameters"]["train"]["semantic_loss_type"] == "CE":
            self.loss_semantic = nn.CrossEntropyLoss(reduction="none", ignore_index=-1)
        elif self.config["hyperparameters"]["train"]["semantic_loss_type"] == "Dice":
            self.loss_semantic = DiceLoss(
                num_encode=self.config["data_flags"]["parameters"]["seg_classes"]
            )
        elif self.config["hyperparameters"]["train"]["semantic_loss_type"] == "FTL":
            self.loss_semantic = FocalTverskyLoss(
                alpha=self.config["hyperparameters"]["train"]["weight_alpha"],
                gamma=self.config["hyperparameters"]["train"]["weight_gamma"],
                num_encode=self.config["data_flags"]["parameters"]["seg_classes"],
            )
        elif self.config["hyperparameters"]["train"]["semantic_loss_type"] == "Dice_CE":
            self.loss_semanticCE = nn.CrossEntropyLoss(
                reduction="none", ignore_index=-1
            )
            self.loss_semanticOther = DiceLoss(
                num_encode=self.config["data_flags"]["parameters"]["seg_classes"]
            )
        elif self.config["hyperparameters"]["train"]["semantic_loss_type"] == "FTL_CE":
            self.loss_semanticCE = nn.CrossEntropyLoss(
                reduction="none", ignore_index=-1
            )
            self.loss_semanticOther = FocalTverskyLoss(
                alpha=self.config["hyperparameters"]["train"]["weight_alpha"],
                gamma=self.config["hyperparameters"]["train"]["weight_gamma"],
                num_encode=self.config["data_flags"]["parameters"]["seg_classes"],
            )

    def step(self, data):
        image = data["input_image"].to(self.config["device"])
        semantic = data["original_seg"].to(self.config["device"])
        semantic = semantic.squeeze().long()

        self.optimizer.zero_grad()

        # predicting semantics from the model
        pred_semantic = self.model(image)

        # calculating one of the following losses depending on parameter provided in config:
        # cross-entropy, dice, focal tversky loss (ftl), combination of cross-entropy and dice/ftl
        if (
            self.config["hyperparameters"]["train"]["semantic_loss_type"] == "Dice_CE"
        ) or (
            self.config["hyperparameters"]["train"]["semantic_loss_type"] == "FTL_CE"
        ):
            loss_semanticCE = self.loss_semanticCE(pred_semantic, semantic)
            loss_semanticCE = self.nan_reduction(loss_semanticCE)
            loss_semanticOther = self.loss_semanticOther(pred_semantic, semantic)
            loss_semantic = (
                loss_semanticCE
                * self.config["hyperparameters"]["train"]["weight_lambda"]
            ) + (
                loss_semanticOther
                * (1 - self.config["hyperparameters"]["train"]["weight_lambda"])
            )
        else:
            loss_semantic = self.loss_semantic(pred_semantic, semantic)

        loss_semantic = self.nan_reduction(loss_semantic)
        loss = loss_semantic

        metrics_seg = seg_metrics(pred_semantic, semantic, self.epsilon, self.config)

        full_metrics = {
            "loss": loss.item(),
            "loss_semantic": loss_semantic.item(),
            **metrics_seg,
        }

        return loss, full_metrics

    def flag_sanity_check(self, flags):
        if flags["type"] is not None:
            raise ValueError("Cannot set flag while running multi-loss")
