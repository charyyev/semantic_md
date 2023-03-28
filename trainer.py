# dataloader files
from datasets.nyu_dataset import NyuDataset
from datasets.hypersim_dataset_v2 import HyperSimDataset
# model file
from models.model_factory import ModelFactory
#loss function file
from utils.loss_functions import BerHuLoss
# Evaluation metrics file
from utils.eval_metrics import depth_metrics

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.nn.functional as F
import torch
import os
import time
from tqdm import tqdm
from collections import defaultdict


class Trainer():
    def __init__(self, config):
        self.config = config
        self.device = config["device"]
        self.prev_val_loss = 1e6

    def prepare_loaders(self):
        aug_config = self.config["augmentation"]

        # train_dataset = NyuDataset(self.config["train"]["data"], self.config["data_location"])
        # self.train_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.config["train"]["batch_size"])
        #
        # val_dataset = NyuDataset(self.config["val"]["data"], self.config["data_location"])
        # self.val_loader = DataLoader(val_dataset, shuffle=False, batch_size=self.config["val"]["batch_size"])

        dataset_root_dir = self.config["data_location"]

        #if using hypersim_dataset_v2 file
        train_dataset = HyperSimDataset(root_dir=dataset_root_dir, train=True, file_path = self.config["train"]["data"], transform=None,
                                        data_flags=self.config["data_flags"])
        self.train_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.config["train"]["batch_size"])

        val_dataset = HyperSimDataset(root_dir=dataset_root_dir, train=False, file_path = self.config["val"]["data"], transform=None,
                                      data_flags=self.config["data_flags"])
        self.val_loader = DataLoader(val_dataset, shuffle=False, batch_size=self.config["val"]["batch_size"])

        # #if using hypersim_dataset file
        # train_dataset = HyperSimDataset(root_dir=dataset_root_dir, train=True, transform=None,
        #                                 data_flags=self.config["data_flags"])
        # self.train_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.config["train"]["batch_size"])

        # val_dataset = HyperSimDataset(root_dir=dataset_root_dir, train=False, transform=None,
        #                               data_flags=self.config["data_flags"])
        # self.val_loader = DataLoader(val_dataset, shuffle=False, batch_size=self.config["val"]["batch_size"])

    def build_model(self):
        learning_rate = self.config["train"]["learning_rate"]
        weight_decay = self.config["train"]["weight_decay"]
        lr_decay_at = self.config["train"]["lr_decay_at"]

        in_channels = 3
        if self.config["data_flags"]["concat"]:
            if self.config["data_flags"]["onehot"]:
                in_channels = 3 + self.config["data_flags"]["seg_classes"]
            else:
                in_channels = 4

        self.model, self.tflags = ModelFactory().get_model(self.config["model"], in_channels=in_channels, classes=1)
        self.model.to(self.device)

        # we have nan values in the target, therefore do not reduce and use self.nan_reduction instead
        self.loss = torch.nn.L1Loss(reduction='none')
        self.nan_reduction = torch.nanmean
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=lr_decay_at, gamma=0.1)

    def train_one_epoch(self, epoch):
        print("training epoch ", epoch)
        total_loss = 0
        total_metrics = defaultdict(float)

        start_time = time.time()
        self.model.train()
        for data in tqdm(self.train_loader):
            image = data["image"].to(self.device)
            target = data["depths"].to(self.device)

            self.optimizer.zero_grad()

            pred = self.model(image)
            loss = self.loss(pred, target)
            loss = self.nan_reduction(loss)
            metrics = depth_metrics(pred, target)

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
            print(f"{k}: {v/len(self.train_loader):.5f}")

    def train(self):
        print("preparing dataloaders...")
        self.prepare_loaders()
        print("building model...")
        self.build_model()
        print("creating experiment log directories...")
        self.make_experiments_dirs()
        self.writer = SummaryWriter(log_dir=self.runs_dir)

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
            for data in self.val_loader:
                image = data["image"].to(self.device)
                target = data["depths"].to(self.device)

                pred = self.model(image)
                loss = self.loss(pred, target)
                loss = self.nan_reduction(loss)
                metrics = depth_metrics(pred, target)

                total_loss += loss.item()
                for k in metrics.keys():
                    total_metrics[k] += metrics[k]

        self.model.train()
        self.writer.add_scalar("val_loss", total_loss / len(self.val_loader), epoch)
        self.writer.add_scalar("val_delta1", total_metrics["delta1"] / len(self.train_loader), epoch)
        self.writer.add_scalar("val_delta2", total_metrics["delta2"] / len(self.train_loader), epoch)
        self.writer.add_scalar("val_delta3", total_metrics["delta3"] / len(self.train_loader), epoch)
        self.writer.add_scalar("val_abs_rel", total_metrics["abs_rel"] / len(self.train_loader), epoch)
        self.writer.add_scalar("val_rmse", total_metrics["rmse"] / len(self.train_loader), epoch)
        self.writer.add_scalar("val_log10", total_metrics["log10"] / len(self.train_loader), epoch)

        print("\nEpoch {} | Time {} | Validation Loss: {:.5f}".format(
            epoch, time.time() - start_time, total_loss / len(self.val_loader)))
        for k, v in total_metrics.items():
            print(f"{k}: {v/len(self.train_loader):.5f}")

        if total_loss / len(self.val_loader) < self.prev_val_loss:
            self.prev_val_loss = total_loss / len(self.val_loader)
            path = os.path.join(self.best_checkpoints_dir, str(epoch) + "epoch")
            torch.save(self.model.state_dict(), path)

    def make_experiments_dirs(self):
        base = self.config["model"] + "_" + self.config["note"] + "_" + self.config["date"] + "_" + str(
            self.config["ver"])
        path = os.path.join(self.config["experiments"], base)
        if not os.path.exists(path):
            os.mkdir(path)
        self.checkpoints_dir = os.path.join(path, "checkpoints")
        self.best_checkpoints_dir = os.path.join(path, "best_checkpoints")
        self.runs_dir = os.path.join(path, "runs")

        if not os.path.exists(self.checkpoints_dir):
            os.mkdir(self.checkpoints_dir)

        if not os.path.exists(self.best_checkpoints_dir):
            os.mkdir(self.best_checkpoints_dir)

        if not os.path.exists(self.runs_dir):
            os.mkdir(self.runs_dir)
