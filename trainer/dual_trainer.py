from trainer import base_trainer

from torch import nn

from utils.eval_metrics import depth_metrics


class DualTrainer(base_trainer.BaseTrainer):
    def build_model(self):
        super().build_model()
        self.loss_depth = nn.L1Loss(reduction="none")
        self.loss_semantic = nn.CrossEntropyLoss(reduction="none")

    def step(self, data):
        image = data["image"].to(self.device)
        depth = data["depths"].to(self.device)
        semantic = data["segs"].to(self.device)

        self.optimizer.zero_grad()

        pred_depth, pred_semantic = self.model(image)
        # clamp values to >0
        loss_depth = self.loss_depth(pred_depth, depth)
        loss_depth = self.nan_reduction(loss_depth)
        loss_semantic = self.loss_semantic(pred_semantic, semantic)
        loss_semantic = self.nan_reduction(loss_semantic)

        lam = self.config["hyperparameters"]["train"]["lambda_loss"]
        loss = loss_depth + lam + loss_semantic

        metrics = depth_metrics(pred_depth, depth, self.epsilon, self.config)

        full_metrics = {
            "loss": loss.item(),
            "loss_depth": loss_depth.item(),
            "loss_semantic": loss_semantic.item(),
            **metrics
        }

        return loss, full_metrics