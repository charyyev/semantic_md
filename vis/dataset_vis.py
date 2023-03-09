import numpy as np

import vispy
from vispy.scene import SceneCanvas
from vispy import app

from datasets.nyu_dataset import NyuDataset


class Vis():
    def __init__(self, dataset):
        self.index = 0
        self.dataset = dataset

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
        image = data["image"].permute((1, 2, 0))
        depths = data["depths"].squeeze().numpy()
        
        image = image.numpy() / 255
        img_depths = (depths - np.min(depths)) / np.max(depths)
        img_depths =  np.repeat(np.expand_dims(img_depths, axis = 2), 3, axis = 2)
        
        
        img = np.concatenate((image, img_depths), 1)       

           
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
    data_file = "/home/sapar/3dvision/data/list/train.txt"
    data_location = "/home/sapar/3dvision/data/nyu_depth_v2_labeled.mat"

    dataset = NyuDataset(data_file, data_location)
    
    vis = Vis(dataset)
    vis.run()