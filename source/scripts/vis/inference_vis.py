import numpy as np
import torch

import vispy
from vispy import app
from vispy.scene import SceneCanvas

from source.datasets.nyu_dataset import NyuDataset
from source.models.unet import Unet


class Vis:
    def __init__(self, dataset, model):
        self.index = 0
        self.dataset = dataset
        self.model = model

        self.canvas = SceneCanvas(keys="interactive", show=True, size=(1920, 480))
        self.canvas.events.key_press.connect(self._key_press)
        self.canvas.events.draw.connect(self._draw)
        self.canvas.show()

        self.view = self.canvas.central_widget.add_view()
        self.image = vispy.scene.visuals.Image(parent=self.view.scene)

        self.update_image()

    def update_image(self):
        data = self.dataset[self.index]
        image = data["image"].unsqueeze(0)
        pred = self.model(image).detach()

        image = image.squeeze().permute((1, 2, 0))
        depths = data["depths"].squeeze().numpy()
        pred = pred.squeeze().numpy()

        image = image.numpy() / 255
        img_depths = (depths - np.min(depths)) / np.max(depths)
        img_depths = np.repeat(np.expand_dims(img_depths, axis=2), 3, axis=2)

        img_pred = (pred - np.min(pred)) / np.max(pred)
        img_pred = np.repeat(np.expand_dims(img_pred, axis=2), 3, axis=2)

        img = np.concatenate((image, img_pred, img_depths), 1)

        self.canvas.title = str(self.index)
        self.image.set_data(img)
        self.view.camera = vispy.scene.PanZoomCamera(aspect=1)
        self.view.camera.set_range()
        self.view.camera.flip = (0, 1, 0)
        self.canvas.update()

    def _key_press(self, event):
        if event.key == "Right":
            if self.index < len(self.dataset) - 1:
                self.index += 1
            self.update_image()

        if event.key == "Left":
            if self.index > 0:
                self.index -= 1
            self.update_image()

        if event.key == "Q":
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
    data_file = "/home/sapar/3dvision/data/list/overfit.txt"
    data_location = "/home/sapar/3dvision/data/nyu_depth_v2_labeled.mat"

    dataset = NyuDataset(data_file, data_location)

    model_path = "/home/sapar/experiments/unet_overfit_09-03-2023_1/checkpoints/95epoch"
    model = Unet(3)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    vis = Vis(dataset, model)
    vis.run()
