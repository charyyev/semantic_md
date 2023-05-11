"""
File is used for training the actual model.
"""
import json
import os
from abc import abstractmethod
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import wandb
from datasets import hypersim_dataset
from models import ModelFactory
from tqdm import tqdm
from utils.eval_metrics import depth_metrics
from utils.logs import ProjectLogger
from utils.loss_functions import BerHuLoss


class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.logger = ProjectLogger(self.config)
        self.prev_val_loss = np.infty
        self.epsilon = 1e-4

        self.build_model()
        self.prepare_loaders()
        self._make_output_dir()
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)

        if self.config["wandb"]:
            self.run = wandb.init(
                project=self.config["project_name"],
                config=config.get_config(),
                dir=self.wandb_dir,
                id=self.id,
            )

            print()
        self.flag_sanity_check(self.config["data_flags"])

    def prepare_loaders(self):
        self.logger.info("Preparing dataloaders")
        (
            image_transform,
            depth_transform,
            seg_transform,
        ) = hypersim_dataset.compute_transforms(self.transform_config, self.config)

        data_dir = self.config.get_subpath("data_location")
        train_file_path = self.config.get_subpath("train_data")
        val_file_path = self.config.get_subpath("val_data")

        train_dataset = hypersim_dataset.HyperSimDataset(
            data_dir=data_dir,
            file_path=train_file_path,
            image_transform=image_transform,
            depth_transform=depth_transform,
            seg_transform=seg_transform,
            data_flags=self.config["data_flags"],
        )
        self.train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.config["hyperparameters"]["train"]["batch_size"],
        )

        val_dataset = hypersim_dataset.HyperSimDataset(
            data_dir=data_dir,
            file_path=val_file_path,
            image_transform=image_transform,
            depth_transform=depth_transform,
            seg_transform=seg_transform,
            data_flags=self.config["data_flags"],
        )
        self.val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=self.config["hyperparameters"]["val"]["batch_size"],
        )

    def build_model(self):
        self.logger.info("Building model")
        learning_rate = self.config["hyperparameters"]["train"]["learning_rate"]
        weight_decay = self.config["hyperparameters"]["train"]["weight_decay"]
        epochs = self.config["hyperparameters"]["train"]["epochs"]

        self.model, self.transform_config = ModelFactory().get_model(
            self.config, in_channels=3
        )
        self.model.to(self.config["device"])

        # self.loss = torch.nn.L1Loss(reduction="none")
        # self.loss = torch.nn.SmoothL1Loss(reduction='none')
        self.loss = BerHuLoss(contains_nan=True)
        self.nan_reduction = torch.nanmean
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=0
        )

    def step(self, data):
        image = data["input_image"].to(self.config["device"])
        target = data["depths"].to(self.config["device"])

        self.optimizer.zero_grad()

        pred = self.model(image)

        loss = self.loss(pred, target)
        loss = self.nan_reduction(loss)
        metrics = depth_metrics(pred, target, self.epsilon, self.config)

        # print(loss.item())

        full_metrics = {"depth_loss": loss.item(), **metrics}

        return loss, full_metrics

    def train_one_epoch(self, epoch):
        self.logger.info(
            f"Training epoch {epoch} / "
            f"{self.config['hyperparameters']['train']['epochs']}"
        )

        total_metrics = defaultdict(float)

        self.model.train()
        for data in tqdm(self.train_loader):
            loss, metrics = self.step(data)

            loss.backward()
            self.optimizer.step()

            for k in metrics.keys():
                total_metrics[k] += metrics[k]

        self.scheduler.step()

        total_metrics = {
            f"train_{k}": v / len(self.train_loader) for k, v in total_metrics.items()
        }
        return total_metrics

        # img = data["original_image"]
        # img = img.clone().detach().cpu().numpy()[0].transpose(1, 2, 0)
        # plt.imshow(img)
        # plt.show()
        # plt.imshow(target.clone().detach().cpu().numpy()[0].transpose(1, 2, 0))
        # plt.show()
        # plt.imshow(pred.clone().detach().cpu().numpy()[0].transpose(1, 2, 0))
        # plt.show()

    def train(self):
        # so logging starts at 1
        if self.config["wandb"]:
            wandb.log({})
        self.logger.info(str(self.config))

        self.logger.info("Saving config file")
        config_save_path = os.path.join(self.output_dir, "config.json")
        with open(config_save_path, "w", encoding="UTF-8") as file:
            json.dump(self.config.get_config(), file)

        if self.config["resume_training"]:
            path = self.config.get_subpath("resume_from")
            self.logger.info(f"Resuming training from {path}")
            self._load_state(path)

        for epoch in range(1, self.config["hyperparameters"]["train"]["epochs"] + 1):
            train_metrics = self.train_one_epoch(epoch)

            if epoch % self.config["hyperparameters"]["train"]["save_every"] == 0:
                path = os.path.join(self.checkpoints_dir, f"epoch_{epoch}")
                self._save_state(path)

            if epoch % self.config["hyperparameters"]["val"]["val_every"] == 0:
                val_metrics = self.validate(epoch)
                self._log(epoch, **train_metrics, **val_metrics)
            else:
                self._log(epoch, **train_metrics)

        self._close()

    def validate(self, epoch):
        self.logger.info(f"Validating: Epoch {epoch}")
        total_loss = 0
        total_metrics = defaultdict(float)

        self.model.eval()
        with torch.no_grad():
            for data in tqdm(self.train_loader):
                _, metrics = self.step(data)

                for k in metrics.keys():
                    total_metrics[k] += metrics[k]

        if total_loss / len(self.val_loader) < self.prev_val_loss:
            self.prev_val_loss = total_loss / len(self.val_loader)
            path = os.path.join(self.best_checkpoints_dir, f"epoch_{epoch}")
            self._save_state(path)

        val_metrics = {"loss": total_loss, **total_metrics}
        val_metrics = {
            f"val_{k}": v / len(self.val_loader) for k, v in val_metrics.items()
        }
        return val_metrics

    def _make_output_dir(self):
        self.logger.info("Creating experiment log directories")
        experiments_folder = self.config.get_subpath("output")
        self.id = name_stem = self.config.get_name_stem()
        self.output_dir = os.path.join(experiments_folder, name_stem)

        os.makedirs(experiments_folder, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=False)

        self.checkpoints_dir = os.path.join(self.output_dir, "checkpoints")
        self.best_checkpoints_dir = os.path.join(self.output_dir, "best_checkpoints")
        self.tensorboard_dir = os.path.join(self.output_dir, "tensorboard")

        os.makedirs(self.checkpoints_dir, exist_ok=False)
        os.makedirs(self.best_checkpoints_dir, exist_ok=False)
        os.makedirs(self.tensorboard_dir, exist_ok=False)

        if self.config["wandb"]:
            self.wandb_dir = os.path.join(self.output_dir, "wandb")
            os.makedirs(self.wandb_dir, exist_ok=False)

    def _save_state(self, path):
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, self.config["save_names"]["weights"])
        optimizer_path = os.path.join(path, self.config["save_names"]["optimizer"])

        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)

    def _load_state(self, path):
        model_path = os.path.join(path, self.config["save_names"]["weights"])
        optimizer_path = os.path.join(path, self.config["save_names"]["optimizer"])

        model_state_dict = torch.load(model_path, map_location=self.config["device"])
        optimizer_state_dict = torch.load(
            optimizer_path, map_location=self.config["device"]
        )

        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)

    def _log(self, epoch: int, **log_dict):
        for name, value in log_dict.items():
            self.writer.add_scalar(name, value, epoch)
            self.logger.info(f"{name}: {value:.5f}")
        if self.config["wandb"]:
            wandb.log(log_dict)

    def _close(self):
        if self.config["wandb"]:
            wandb.finish(exit_code=0, quiet=False)
        # sys.exit(0)

    @abstractmethod
    def flag_sanity_check(self, flags):
        """
        Checks the flags for compatibility
        """
        id(flags)
