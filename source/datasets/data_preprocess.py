import os
import sys

sys.path.insert(0, "/media/ankitaghosh/Data/ETH/3DVision/semantic_md/")
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

import cv2
import h5py
from tqdm import tqdm

from source.utils.conversions import (
    semantic_norm,
    semantic_to_border,
    simplified_encode_3,
    simplified_encode_4,
)

"""
HyperSim_Data
|
├── image
│   ├── ai_001_001                                  <--- folder which corresponds to downloaded zip file
│   │   ├── images
│   │   │   ├── scene_cam_00_final_hdf5             <--- this is where the images are stored
│   │   │   │   └── frame.0000.color.hdf5           <--- example image
│   │   └── ...
│   │
├── depth
│   ├── ai_001_001                                  <--- folder which corresponds to downloaded zip file
│   │   ├── images
│   │   │   ├── scene_cam_00_geometry_hdf5          <--- this is where the labels are stored
│   │   │   │   ├── frame.0000.depth_meters.hdf5    <--- example depth map
│   │   └── ...
│   │
├── semantic
    ├── ai_001_001                                  <--- folder which corresponds to downloaded zip file
        ├── images
        │   ├── scene_cam_00_geometry_hdf5          <--- this is where the labels are stored
        │   │   ├── frame.0000.semantic.hdf5        <--- example semantic map
        └── ...

Image Path = ROOTDIR/HyperSim_Data/image/ai_xxx_xxx/images/scene_cam_00_final_hdf5/frame.yyyy.color.hdf5

Depth Path = ROOTDIR/HyperSim_Data/depth/ai_xxx_xxx/images/cene_cam_00_geometry_hdf5/frame.yyyy.depth_meters.hdf5
replace-- /image/ with /depth/ ; _final_hdf5 with _geometry_hdf5 ; color.hdf5 with depth_meters.hdf5

Semantic Path = ROOTDIR/HyperSim_Data/semantic/ai_xxx_xxx/images/cene_cam_00_geometry_hdf5/frame.yyyy.semantic.hdf5
replace-- /image/ with /semantic/ ; _final_hdf5 with _geometry_hdf5 ; color.hdf5 with semantic.hdf5

"""


class HyperSimDataset(Dataset):
    def __init__(
        self,
        data_dir,
        file_path="",
        image_transform=None,
        depth_transform=None,
        seg_transform=None,
        data_flags=None,
    ):
        """
        Dataset class for HyperSim
        :param data_dir: the root directory of the dataset, which contains the uncompressed data
        :param train: train or test set
        :param test_split: percentage of dataset to use as test set [0, 1]
        :param transform: transform functions
        :param data_flags: "concat" to additionally get image+seg concatenated, "onehot" for one-hot encoding of seg
        """
        self.random_seed = 0
        self.data_dir = data_dir
        self.image_transform = image_transform
        self.depth_transform = depth_transform
        self.seg_transform = seg_transform
        self.data_flags = data_flags

        if self.image_transform is None:
            self.image_transform = transforms.ToTensor()
        if self.depth_transform is None:
            self.depth_transform = transforms.ToTensor()
        if self.seg_transform is None:
            self.seg_transform = transforms.ToTensor()

        if self.data_flags is None:
            self.data_flags = {}

        random.seed(self.random_seed)
        np.random.seed()

        image_paths = []
        depth_paths = []
        seg_paths = []

        if file_path == "":
            print("File path not given for loading data...")
        else:
            print("Processing data...")
        with open(file_path, "r", encoding="UTF-8") as file:
            for line in file:
                imgPath = os.path.join(self.data_dir, line.replace("\n", ""))
                depthPath = os.path.join(
                    self.data_dir,
                    imgPath.replace("/image/", "/depth/")
                    .replace("_final_hdf5", "_geometry_hdf5")
                    .replace("color.hdf5", "depth_meters.hdf5"),
                )
                semPath = os.path.join(
                    self.data_dir,
                    imgPath.replace("/image/", "/semantic/")
                    .replace("_final_hdf5", "_geometry_hdf5")
                    .replace("color.hdf5", "semantic.hdf5"),
                )

                if (
                    os.path.exists(imgPath)
                    and os.path.exists(depthPath)
                    and os.path.exists(semPath)
                ):
                    image_paths.append(imgPath)
                    depth_paths.append(depthPath)
                    seg_paths.append(semPath)
                else:
                    print(imgPath)

        # assert length of all is the same
        assert len(image_paths) == len(depth_paths) == len(seg_paths)

        # shuffle all arrays (determined by random seed), relative order stays the same
        self.image_paths = np.array(image_paths)
        self.depth_paths = np.array(depth_paths)
        self.seg_paths = np.array(seg_paths)
        self.paths = np.column_stack(
            (self.image_paths, self.depth_paths, self.seg_paths)
        )

        # Print images with infinity values, TODO: delete
        printPath = []
        for image_path in tqdm(self.seg_paths):
            try:
                with h5py.File(image_path, "r") as image:
                    image_np = np.array(image["dataset"])
                    if np.isinf(image_np).any():
                        printPath.append(image_path)
            except:
                print("Error: unable to open", image_path)
        for i in range(0, len(printPath)):
            print(printPath[i])

        self.length = self.paths.shape[0]


def compute_transforms(transform_config, config):
    tcfg = transform_config
    mean, std = tcfg["mean"], tcfg["std"]
    depth = config["transformations"]["depth_range"]
    min_depth, max_depth = depth["min"], depth["max"]
    new_size = config["transformations"]["resize"]
    new_width, new_height = new_size["width"], new_size["height"]

    def resize(input_):
        return cv2.resize(
            input_, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )

    base_transform = (transforms.ToTensor(),)

    def image_transform(input_):
        x = resize(input_)
        tf = transforms.Compose([*base_transform, transforms.Normalize(mean, std)])
        return tf(x)

    def depth_transform(input_):
        x = resize(input_)
        x = np.clip(x, min_depth, max_depth)
        x = (x - min_depth) / (max_depth - min_depth)
        tf = transforms.Compose([*base_transform])
        return tf(x)

    def seg_transform(input_):
        x = resize(input_)
        tf = transforms.Compose([*base_transform])
        return tf(x)

    return image_transform, depth_transform, seg_transform


def test():
    ROOT_DIR = "/media/ankitaghosh/Data/ETH/3DVision/HyperSim_trial/"
    TXT_FILE = "/media/ankitaghosh/Data/ETH/3DVision/HyperSim_trial/newData.txt"
    HyperSimDataset(data_dir=ROOT_DIR, file_path=TXT_FILE)


if __name__ == "__main__":
    test()
