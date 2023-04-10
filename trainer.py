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


class Trainer():
    def __init__(self, config):
        self.config = config
        self.device = config["device"]
        self.prev_val_loss = 1e6
        self.epsilon = 1e-4

        self._data_flag_sanity_check()

    def prepare_loaders(self):
        # aug_config = self.config["augmentation"]

        dataset_root_dir = self.config["data_location"]
        image_transform, depth_transform, seg_transform = compute_transforms(self.transform_config, self.config)

        # #if using hypersim_dataset file
        train_dataset = dataset.HyperSimDataset(root_dir=dataset_root_dir, file_path=self.config["train"]["data"],
                                                image_transform=image_transform, depth_transform=depth_transform,
                                                seg_transform=seg_transform, data_flags=self.config["data_flags"])
        self.train_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.config["train"]["batch_size"])

        val_dataset = dataset.HyperSimDataset(root_dir=dataset_root_dir, file_path=self.config["val"]["data"],
                                              image_transform=image_transform, depth_transform=depth_transform,
                                              seg_transform=seg_transform, data_flags=self.config["data_flags"])
        self.val_loader = DataLoader(val_dataset, shuffle=False, batch_size=self.config["val"]["batch_size"])

    def build_model(self):
        learning_rate = self.config["train"]["learning_rate"]
        weight_decay = self.config["train"]["weight_decay"]
        epochs = self.config["train"]["epochs"]

        in_channels = 3
        if self.config["data_flags"]["concat"]:
            if self.config["data_flags"]["onehot"]:
                in_channels = 3 + self.config["data_flags"]["seg_classes"]
            else:
                in_channels = 4

        pretrained_weights_path = os.path.join(self.config["root_dir"], "models", "pretrained_weights")
        self.model, self.transform_config = ModelFactory() \
            .get_model(self.config["model"], pretrained_weights_path, in_channels=in_channels,
                       semantic_convolution=self.config["data_flags"]["semantic_convolution"])
        self.model.to(self.device)

        # we have nan values in the target, therefore do not reduce and use self.nan_reduction instead
        self.loss = torch.nn.L1Loss(reduction='none')
        # self.loss = torch.nn.SmoothL1Loss(reduction='none')
        # self.loss = BerHuLoss(contains_nan=True)
        self.nan_reduction = torch.nanmean
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=0)

    def train_one_epoch(self, epoch):
        print("training epoch ", epoch)
        total_loss = 0
        total_metrics = defaultdict(float)

        start_time = time.time()
        self.model.train()
        for data in tqdm(self.train_loader):
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

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            for k in metrics.keys():
                total_metrics[k] += metrics[k]
        self.writer.add_scalar("train_loss", total_loss / len(self.train_loader), epoch)
        self.writer.add_scalar("train_delta1", total_metrics["delta1"] / len(self.train_loader), epoch)
        self.writer.add_scalar("train_delta2", total_metrics["delta2"] / len(self.train_loader), epoch)
        self.writer.add_scalar("train_delta3", total_metrics["delta3"] / len(self.train_loader), epoch)
        self.writer.add_scalar("train_abs_rel", total_metrics["abs_rel"] / len(self.train_loader), epoch)
        self.writer.add_scalar("train_rmse", total_metrics["rmse"] / len(self.train_loader), epoch)
        self.writer.add_scalar("train_log10", total_metrics["log10"] / len(self.train_loader), epoch)

        print("\nEpoch {} | Time {}| Training Loss: {:.5f}".format(
            epoch, time.time() - start_time, total_loss / len(self.train_loader)))
        for k, v in total_metrics.items():
            print(f"{k}: {v / len(self.train_loader):.5f}")

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
            model_path = os.path.join(self.checkpoints_dir, str(self.config["resume_from"]) + "epoch")
            self.model.load_state_dict(torch.load(model_path, map_location=self.config["device"]))
            start_epoch = self.config["resume_from"]
            print("successfully loaded model starting from " + str(self.config["resume_from"]) + " epoch")

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
                image = data["image"].to(self.device)
                target = data["depths"].to(self.device)

                pred = self.model(image)
                loss = self.loss(pred, target)
                loss = self.nan_reduction(loss)
                metrics = depth_metrics(pred, target, self.epsilon, self.config)

                # print(loss.item())
                total_loss += loss.item()
                for k in metrics.keys():
                    total_metrics[k] += metrics[k]

        self.model.train()
        self.writer.add_scalar("val_loss", total_loss / len(self.val_loader), epoch)
        self.writer.add_scalar("val_delta1", total_metrics["delta1"] / len(self.val_loader), epoch)
        self.writer.add_scalar("val_delta2", total_metrics["delta2"] / len(self.val_loader), epoch)
        self.writer.add_scalar("val_delta3", total_metrics["delta3"] / len(self.val_loader), epoch)
        self.writer.add_scalar("val_abs_rel", total_metrics["abs_rel"] / len(self.val_loader), epoch)
        self.writer.add_scalar("val_rmse", total_metrics["rmse"] / len(self.val_loader), epoch)
        self.writer.add_scalar("val_log10", total_metrics["log10"] / len(self.val_loader), epoch)

        print("\nEpoch {} | Time {} | Validation Loss: {:.5f}".format(
            epoch, time.time() - start_time, total_loss / len(self.val_loader)))
        for k, v in total_metrics.items():
            print(f"{k}: {v / len(self.val_loader):.5f}")

        if total_loss / len(self.val_loader) < self.prev_val_loss:
            self.prev_val_loss = total_loss / len(self.val_loader)
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
        self.tensorboard_dir = os.path.join(self.config["experiments"], "tensorboard", base + str(version))
        self.exp_path = path

        if not os.path.exists(self.checkpoints_dir):
            os.mkdir(self.checkpoints_dir)

        if not os.path.exists(self.best_checkpoints_dir):
            os.mkdir(self.best_checkpoints_dir)

        if not os.path.exists(self.runs_dir):
            os.mkdir(self.runs_dir)
        
        if not os.path.exists(self.tensorboard_dir):
            os.mkdir(self.tensorboard_dir)

    def _data_flag_sanity_check(self):
        """
        Checks the data flags for compatibility. List of compatibilities:
        (1) When semantic_convolution is set, ["concat", "onehot", "border"]
        :return: Error if compatibility checks are failed
        """
        if self.config["data_flags"]["semantic_convolution"]:
            if self.config["data_flags"]["concat"] or self.config["data_flags"]["onehot"]\
                    or self.config["data_flags"]["border"]:
                raise ValueError("'concat' and 'onehot' data_flags cannot be set when 'semantic_convolution' is set.")
