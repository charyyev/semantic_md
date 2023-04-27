from trainer.base_trainer import BaseTrainer
from utils.eval_metrics import depth_metrics


class SemanticConvolutionTrainer(BaseTrainer):
    def step(self, data):
        image = data["image"].to(self.config["device"])
        target = data["depths"].to(self.config["device"])
        semantic = data["segs"].to(self.config["device"])

        self.optimizer.zero_grad()

        pred = self.model(image, semantic)

        loss = self.loss(pred, target)
        loss = self.nan_reduction(loss)
        metrics = depth_metrics(pred, target, self.epsilon, self.config)

        # print(loss.item())

        full_metrics = {"loss": loss.item(), **metrics}

        return loss, full_metrics

    def flag_sanity_check(self, flags):
        id(flags)
