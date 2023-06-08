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
        # initialization
        self.scheduler = None
        self.optimizer = None
        self.nan_reduction = None
        self.transform_config = None
        self.model = None
        self.loss = None
        self.val_loader = None
        self.train_loader = None
        self.config = config
        self.logger = ProjectLogger(self.config)
        self.prev_val_loss = np.infty
        self.epsilon = 1e-4

        self.build_model()
        self.prepare_loaders()
        self._make_output_dir()
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)

        # initialize wandb
        self.logger.debug("initializing wandb")
        if self.config["wandb"]["wandb"]:
            self.run = wandb.init(
                entity=self.config["wandb"]["entity"],
                project=self.config["wandb"]["project"],
                group=self.config["wandb"]["group"],
                config=config.get_config(),
                dir=self.wandb_dir,
                id=self.id,
            )

            print()
        self.logger.debug("wandb done")
        self.logger.debug("doing data flag sanity check")
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
        """
        specifies model, losses, optimizer, schedulers
        """
        self.logger.info("Building model")
        learning_rate = self.config["hyperparameters"]["train"]["learning_rate"]
        weight_decay = self.config["hyperparameters"]["train"]["weight_decay"]
        epochs = self.config["hyperparameters"]["train"]["epochs"]

        # get the model specified in the config
        self.model, self.transform_config = ModelFactory().get_model(
            self.config, in_channels=3
        )
        self.model.to(self.config["device"])

        if self.config["hyperparameters"]["train"]["depth_loss_type"] == "L1":
            self.loss = torch.nn.L1Loss(reduction="none")
        elif self.config["hyperparameters"]["train"]["depth_loss_type"] == "berhu":
            self.loss = BerHuLoss(contains_nan=True)
        else:
            raise ValueError("Please specify correct depth loss")

        # we want to ignore pixels with nan values, which is why we use nanmean for
        # reduction of losses
        self.nan_reduction = torch.nanmean
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

    def step(self, data):
        """
        Does a single step.
        1) gets data
        2) computes prediction
        3) computes losses and metrics
        does explicitly not do gradient descent, as it is used by both training and
        validation
        """
        image = data["input_image"].to(self.config["device"])
        target = data["depths"].to(self.config["device"])

        self.optimizer.zero_grad()

        pred = self.model(image)

        loss = self.loss(pred, target)
        loss = self.nan_reduction(loss)
        metrics = depth_metrics(pred, target, self.epsilon, self.config)

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

        # average metrics over epoch
        total_metrics = {
            f"train_{k}": v / len(self.train_loader) for k, v in total_metrics.items()
        }
        return total_metrics

    def train(self):
        # initialize logging, so it starts at 1
        if self.config["wandb"]["wandb"]:
            wandb.log({})
        self.logger.info(str(self.config))

        self.logger.info("Saving config file")
        config_save_path = os.path.join(self.output_dir, "config.json")
        with open(config_save_path, "w", encoding="UTF-8") as file:
            json.dump(self.config.get_config(), file)

        start_epoch = 1
        if self.config["resume"]["resume_training"]:
            path = self.config["resume"]["path"]
            self.logger.info(f"Resuming training from {path}")
            self._load_state(path)
            start_epoch = self.config["resume"]["epoch"]

        for epoch in range(
            start_epoch, self.config["hyperparameters"]["train"]["epochs"] + 1
        ):
            # 1) train an epoch
            train_metrics = self.train_one_epoch(epoch)

            # 2) save model if necessary (always save last epoch)
            if (
                epoch % self.config["hyperparameters"]["train"]["save_every"] == 0
                or epoch == self.config["hyperparameters"]["train"]["epochs"]
            ):
                path = os.path.join(self.checkpoints_dir, f"epoch_{epoch}")
                self._save_state(path)

            # 3) validate model performance if necessary
            if epoch % self.config["hyperparameters"]["val"]["val_every"] == 0:
                val_metrics = self.validate(epoch)
                self._log(epoch, **train_metrics, **val_metrics)
            else:
                self._log(epoch, **train_metrics)

        self._close()

    def validate(self, epoch):
        """
        Validation. Checks each image in the validation set and averages the results for
        logging.
        """
        self.logger.info(f"Validating: Epoch {epoch}")
        total_loss = 0
        total_metrics = defaultdict(float)

        self.model.eval()
        with torch.no_grad():
            for data in tqdm(self.val_loader):
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
        """
        Creates all the different directories for saving results and metrics.
        """
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

        if self.config["wandb"]["wandb"]:
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
        if self.config["wandb"]["wandb"]:
            wandb.log(log_dict)

    def _close(self):
        if self.config["wandb"]["wandb"]:
            wandb.finish(exit_code=0, quiet=False)
        # sys.exit(0)

    @abstractmethod
    def flag_sanity_check(self, flags):
        """
        Checks the flags for compatibility, overwritten in next classes
        """
        id(flags)
