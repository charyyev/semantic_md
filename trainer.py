from datasets.nyu_dataset import NyuDataset
from datasets.hypersim_dataset import HyperSimDataset
from models.model_factory import ModelFactory

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.nn.functional as F
import torch
import os
import time
from tqdm import tqdm


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

        hypersim_root_dir = self.config["data"]
        tf = transforms.ToTensor()
        train_dataset = HyperSimDataset(root_dir=hypersim_root_dir, train=True, transform=tf)
        self.train_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.config["train"]["batch_size"])

        val_dataset = HyperSimDataset(root_dir=hypersim_root_dir, train=False, transform=tf)
        self.val_loader = DataLoader(val_dataset, shuffle=False, batch_size=self.config["val"]["batch_size"])

    def build_model(self):
        learning_rate = self.config["train"]["learning_rate"]
        weight_decay = self.config["train"]["weight_decay"]
        lr_decay_at = self.config["train"]["lr_decay_at"]

        self.model, self.tflags = ModelFactory().get_model('UResNet', in_channels=3, classes=1357)
        self.model.to(self.device)

        self.loss = torch.nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=lr_decay_at, gamma=0.1)

    def train_one_epoch(self, epoch):
        total_loss = 0

        start_time = time.time()
        self.model.train()
        for data in tqdm(self.train_loader):
            image = data["image"].float().to(self.device)
            target = data["depths"].float().to(self.device)

            self.optimizer.zero_grad()

            pred = self.model(image)
            loss = self.loss(pred, target)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        self.writer.add_scalar("train_loss", total_loss / len(self.train_loader), epoch)

        print("Epoch {}|Time {}|Training Loss: {:.5f}".format(
            epoch, time.time() - start_time, total_loss / len(self.train_loader)))

    def train(self):
        self.prepare_loaders()
        self.build_model()
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
                path = os.path.join(self.checkpoints_dir, str(epoch) + "epoch")
                torch.save(self.model.state_dict(), path)

            if (epoch + 1) % self.config["val"]["val_every"] == 0:
                self.validate(epoch)

            self.scheduler.step()

    def validate(self, epoch):
        total_loss = 0

        start_time = time.time()
        self.model.eval()
        with torch.no_grad():
            for data in self.val_loader:
                image = data["image"].to(self.device)
                target = data["depths"].to(self.device)

                pred = self.model(image)
                loss = self.loss(pred, target)

                total_loss += loss.item()

        self.model.train()
        self.writer.add_scalar("val_loss", total_loss / len(self.val_loader), epoch)

        print("Epoch {}|Time {}|Validation Loss: {:.5f}".format(
            epoch, time.time() - start_time, total_loss / len(self.val_loader)))

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
