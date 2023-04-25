# dataloader files
# import matplotlib.pyplot as plt

from datasets import hypersim_dataset as dataset
# model file
from models.model_factory import ModelFactory
# loss function file
from utils.loss_functions import BerHuLoss
# Evaluation metrics file
from utils.eval_metrics import depth_metrics
# Transformation
from utils.transforms import compute_transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import time
from tqdm import tqdm
from collections import defaultdict
import json


class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config["device"]
        self.prev_val_loss = 1e6
        self.epsilon = 1e-4

        data_flag_sanity_check(self.config["data_flags"])

    def prepare_loaders(self):
        # aug_config = self.config["augmentation"]

        dataset_root_dir = self.config["data_location"]
        image_transform, depth_transform, seg_transform = compute_transforms(
            self.transform_config, self.config)

        # #if using hypersim_dataset file
        train_dataset = dataset.HyperSimDataset(root_dir=dataset_root_dir,
                                                file_path=self.config["train"]["data"],
                                                image_transform=image_transform,
                                                depth_transform=depth_transform,
                                                seg_transform=seg_transform,
                                                data_flags=self.config["data_flags"])
        self.train_loader = DataLoader(train_dataset, shuffle=True,
                                       batch_size=self.config["train"]["batch_size"])

        val_dataset = dataset.HyperSimDataset(root_dir=dataset_root_dir,
                                              file_path=self.config["val"]["data"],
                                              image_transform=image_transform,
                                              depth_transform=depth_transform,
                                              seg_transform=seg_transform,
                                              data_flags=self.config["data_flags"])
        self.val_loader = DataLoader(val_dataset, shuffle=False,
                                     batch_size=self.config["val"]["batch_size"])

    def build_model(self):
        learning_rate = self.config["train"]["learning_rate"]
        weight_decay = self.config["train"]["weight_decay"]
        epochs = self.config["train"]["epochs"]

        pretrained_weights_path = os.path.join(self.config["root_dir"], "models",
                                               "pretrained_weights")
        self.model, self.transform_config = ModelFactory() \
            .get_model(self.config["model"], pretrained_weights_path, self.config,
                       in_channels=3)
        self.model.to(self.device)

        # we have nan values in the target, therefore do not reduce and use self.nan_reduction instead
        self.loss = torch.nn.L1Loss(reduction='none')
        # self.loss = torch.nn.SmoothL1Loss(reduction='none')
        # self.loss = BerHuLoss(contains_nan=True)
        self.nan_reduction = torch.nanmean
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate,
                                          weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                    T_max=epochs,
                                                                    eta_min=0)

    def step(self, data):
        image = data["image"].to(self.device)
        target = data["depths"].to(self.device)
        semantic = data["segs"].to(self.device)

        self.optimizer.zero_grad()

        if self.config["data_flags"]["semantic_convolution"]:
            pred = self.model(image, semantic)
        else:
            pred = self.model(image)
        # clamp values to >0
        loss = self.loss(pred, target)
        loss = self.nan_reduction(loss)
        metrics = depth_metrics(pred, target, self.epsilon, self.config)

        # print(loss.item())

        full_metrics = {"loss": loss.item(), **metrics}

        return loss, full_metrics

    def train_one_epoch(self, epoch):
        print("training epoch ", epoch)
        total_loss = 0
        total_metrics = defaultdict(float)

        start_time = time.time()
        self.model.train()
        for data in tqdm(self.train_loader):
            loss, metrics = self.step(data)

            loss.backward()
            self.optimizer.step()

            for k in metrics.keys():
                total_metrics[k] += metrics[k]

        total_metrics = {f"train_{k}": v / len(self.train_loader) for k, v in
                         total_metrics}
        self._log(total_metrics, epoch=epoch)

        print("\nEpoch {} | Time {}| Training Loss: {:.5f}".format(
            epoch, time.time() - start_time, total_metrics["train_loss"]))
        for k, v in total_metrics.items():
            print(f"{k}: {v:.5f}")

        # img = data["original_image"]
        # img = img.clone().detach().cpu().numpy()[0].transpose(1, 2, 0)
        # plt.imshow(img)
        # plt.show()
        # plt.imshow(target.clone().detach().cpu().numpy()[0].transpose(1, 2, 0))
        # plt.show()
        # plt.imshow(pred.clone().detach().cpu().numpy()[0].transpose(1, 2, 0))
        # plt.show()

    def train(self):
        print("building model...")
        self.build_model()
        print("preparing dataloaders...")
        self.prepare_loaders()
        print("creating experiment log directories...")
        self.make_experiments_dirs()
        self.writer = SummaryWriter(log_dir=self.runs_dir)
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)

        # save config file
        if not os.path.exists(os.path.join(self.exp_path, "config.json")):
            config_json = json.dumps(self.config, indent=4)
            with open(os.path.join(self.exp_path, "config.json"), 'w') as outfile:
                outfile.write(config_json)

        start_epoch = 0
        if self.config["resume_training"]:
            model_path = os.path.join(self.checkpoints_dir,
                                      str(self.config["resume_from"]) + "epoch")
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.config["device"]))
            start_epoch = self.config["resume_from"]
            print("successfully loaded model starting from " + str(
                self.config["resume_from"]) + " epoch")

        for epoch in range(start_epoch + 1, self.config["train"]["epochs"]):
            self.train_one_epoch(epoch)

            if epoch % self.config["train"]["save_every"] == 0:
                path = os.path.join(self.checkpoints_dir, str(epoch) + "epoch.pth")
                torch.save(self.model.state_dict(), path)

            if (epoch + 1) % self.config["val"]["val_every"] == 0:
                self.validate(epoch)

            self.scheduler.step()

    def validate(self, epoch):
        print("validation epoch ", epoch)
        total_loss = 0
        total_metrics = defaultdict(float)

        start_time = time.time()
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(self.val_loader):
                loss, metrics = self.step(data)

                for k in metrics.keys():
                    total_metrics[k] += metrics[k]

        total_metrics = {f"val_{k}": v / len(self.val_loader) for k, v in
                             total_metrics}
        self._log(total_metrics, epoch=epoch)

        self.model.train()

        print("\nEpoch {} | Time {} | Validation Loss: {:.5f}".format(
            epoch, time.time() - start_time, total_metrics["val_loss"]))
        for k, v in total_metrics.items():
            print(f"{k}: {v:.5f}")

        if total_metrics["val_loss"] < self.prev_val_loss:
            self.prev_val_loss = total_metrics["val_loss"]
            path = os.path.join(self.best_checkpoints_dir, str(epoch) + "epoch.pth")
            torch.save(self.model.state_dict(), path)

    def make_experiments_dirs(self):
        model = self.config["model"]
        note = self.config["note"]
        date = self.config["date"]
        base = f"{model}_{date}_{note}_v"

        version = 0
        while True:
            path = os.path.join(self.config["experiments"], base + str(version))
            if os.path.exists(path):
                version += 1
            else:
                os.mkdir(path)
                break
        self.checkpoints_dir = os.path.join(path, "checkpoints")
        self.best_checkpoints_dir = os.path.join(path, "best_checkpoints")
        self.runs_dir = os.path.join(path, "runs")
        self.tensorboard_dir = os.path.join(path, "tensorboard")
        self.exp_path = path

        if not os.path.exists(self.checkpoints_dir):
            os.mkdir(self.checkpoints_dir)

        if not os.path.exists(self.best_checkpoints_dir):
            os.mkdir(self.best_checkpoints_dir)

        if not os.path.exists(self.runs_dir):
            os.mkdir(self.runs_dir)

        if not os.path.exists(self.tensorboard_dir):
            os.mkdir(self.tensorboard_dir)

    def _log(self, metrics, epoch):
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, epoch)


def data_flag_sanity_check(data_flags):
    """
    Checks the data flags for compatibility. List of compatibilities:
    (1) Only one of ["concat", "onehot", "border", "semantic_convolution"]
    :return: Error if compatibility checks are failed
    """
    value_list = [int(v) for (k, v) in data_flags.items() if k not in ["seg_classes"]]
    if sum(value_list) > 1:
        raise ValueError(
            f"Only one of ['concat', 'onehot', 'border', 'semantic_convolution', 'simplified_onehot'] can be true. Is {data_flags}")
