from torch import nn

from trainer.base_trainer import BaseTrainer
from utils.eval_metrics import depth_metrics, seg_metrics



class MultiLossTrainer(BaseTrainer):
    def build_model(self):
        super().build_model()
        if self.config["hyperparameters"]["train"]["depth_loss_type"] == "L1":
            self.loss_depth = nn.L1Loss(reduction="none")
        elif self.config["hyperparameters"]["train"]["depth_loss_type"] == "berhu":
            self.loss_depth = BerHuLoss(contains_nan=True)
        self.loss_semantic = nn.CrossEntropyLoss(reduction="none")

    def step(self, data):
        image = data["input_image"].to(self.config["device"])
        depth = data["depths"].to(self.config["device"])
        semantic = data["input_segs"].to(self.config["device"])
        semantic = semantic.squeeze().long()
        #semantic = semantic.squeeze().long() - 1

        self.optimizer.zero_grad()

        pred_depth, pred_semantic = self.model(image)
        # clamp values to >0
        loss_depth = self.loss_depth(pred_depth, depth)
        loss_depth = self.nan_reduction(loss_depth)
        loss_semantic = self.loss_semantic(pred_semantic, semantic)
        loss_semantic = self.nan_reduction(loss_semantic)

        lam = self.config["hyperparameters"]["train"]["lambda_loss"]
        loss = loss_depth + lam + loss_semantic

        metrics_depth = depth_metrics(pred_depth, depth, self.epsilon, self.config)
        metrics_seg = seg_metrics(pred_semantic, semantic, self.epsilon, self.config)
        

        full_metrics = {
            "loss": loss.item(),
            "loss_depth": loss_depth.item(),
            "loss_semantic": loss_semantic.item(),
            **metrics_depth,
            **metrics_seg
        }

        return loss, full_metrics

    def flag_sanity_check(self, flags):
        if flags["type"] is not None:
            raise ValueError("Cannot set flag while running multi-loss")
