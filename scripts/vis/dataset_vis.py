import json
import numpy as np
import torch
import vispy
from torchvision.transforms import transforms
from vispy import app
from vispy.scene import SceneCanvas

from datasets import hypersim_dataset
from datasets.hypersim_dataset import HyperSimDataset
from models.model_factory import ModelFactory
from utils.config import args_and_config


class Vis():
    def __init__(self, dataset, model, config):
        self.index = 0
        self.dataset = dataset
        self.model = model
        self.config = config

        self.canvas = SceneCanvas(keys='interactive',
                                  show=True,
                                  size=(1280, 480))
        self.canvas.events.key_press.connect(self._key_press)
        self.canvas.events.draw.connect(self._draw)
        self.canvas.show()

        self.view = self.canvas.central_widget.add_view()
        self.image = vispy.scene.visuals.Image(parent=self.view.scene)

        self.update_image()

    def update_image(self):
        data = self.dataset[self.index]
        image = torch.unsqueeze(data["image"], dim=0).to(self.config["device"])
        depths = data["depths"].squeeze().numpy()

        pred = self.model(image)
        pred = torch.squeeze(pred).cpu().detach().numpy()

        image = torch.squeeze(image)
        image = image.permute((1, 2, 0)).cpu().detach().numpy()[:, :, :3]
        img_depths = (depths - np.nanmin(depths)) / np.nanmax(depths)
        img_depths = np.repeat(np.expand_dims(img_depths, axis=2), 3, axis=2)
        img_preds = (pred - np.nanmin(pred)) / np.nanmax(pred)
        img_preds = np.repeat(np.expand_dims(img_preds, axis=2), 3, axis=2)

        img = np.concatenate((image, img_depths, img_preds), 1)

        self.canvas.title = str(self.index)
        self.image.set_data(img)
        self.view.camera = vispy.scene.PanZoomCamera(aspect=1)
        self.view.camera.set_range()
        self.view.camera.flip = (0, 1, 0)
        self.canvas.update()

    def _key_press(self, event):
        if event.key == 'Right':
            if self.index < len(self.dataset) - 1:
                self.index += 1
            self.update_image()

        if event.key == 'Left':
            if self.index > 0:
                self.index -= 1
            self.update_image()

        if event.key == 'Q':
            self.destroy()

    def destroy(self):
        # destroy the visualization
        self.canvas.close()
        vispy.app.quit()

    def _draw(self, event):
        if self.canvas.events.key_press.blocked():
            self.canvas.events.key_press.unblock()

    def run(self):
        self.canvas.app.run()


if __name__ == "__main__":
    config = args_and_config()

    dataset_root_dir = config["data_location"]
    data_flags = config["data_flags"]

    mean, std = hypersim_dataset.get_normalizers(train=True)
    image_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    depth_transform = transforms.Compose([transforms.ToTensor()])
    seg_transform = transforms.Compose([transforms.ToTensor()])
    dataset = hypersim_dataset.HyperSimDataset(root_dir=dataset_root_dir, train=False,
                                               image_transform=image_transform,
                                               depth_transform=depth_transform, seg_transform=seg_transform,
                                               data_flags=config["data_flags"])

    model_path = config["load"]["path"]
    in_channels = 3
    if config["data_flags"]["concat"]:
        if config["data_flags"]["onehot"]:
            in_channels = 3 + config["data_flags"]["seg_classes"]
        else:
            in_channels = 4

    model, tflags = ModelFactory().get_model(config["model"], in_channels=in_channels, classes=1)
    model.to(config["device"])
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()

    vis = Vis(dataset, model, config)
    vis.run()