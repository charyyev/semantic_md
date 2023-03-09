import numpy as np
import os
import h5py
import tifffile
import cv2

import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as T
from torch.utils.data import DataLoader

from utils.transforms import Random_Rotation, OneOf, Affine

class NyuDataset(Dataset):
    def __init__(self, data_file, data_location):
        self.data_file = data_file
        self.path = data_location

        self.create_data_list()
        
        file = h5py.File(self.path, mode='r')
        self.images = file["images"]
        self.depths = file["depths"]
        #self.labels = file["labels"]
        
        # file.close()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        index = self.data_list[idx]
        
        image = torch.from_numpy(self.images[index])
        depths = torch.from_numpy(self.depths[index])

        image = image.permute(0, 2, 1).type(torch.float32)
        depths = depths.permute(1, 0).unsqueeze(0).type(torch.float32)
        
        return {"image": image, 
                "depths": depths}


    def create_data_list(self):
        data_list = []
        with open(self.data_file, "r") as f:
            for line in f:
                data_list.append(int(line.strip()))
        
        self.data_list = data_list



if __name__ == "__main__":
    data_file = "/home/sapar/3dvision/data/list/train.txt"
    data_location = "/home/sapar/3dvision/data/nyu_depth_v2_labeled.mat"
    dataset = NyuDataset(data_file, data_location)
    data_loader = DataLoader(dataset, shuffle=True, batch_size=1)

    for data in data_loader:
        image = data["image"]
        depths = data["depths"]

        print(image.shape)
        print(depths.shape)
        break