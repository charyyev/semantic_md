import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

import cv2
from utils.conversions import (
    semantic_encode,
    semantic_norm,
    semantic_to_border,
    simplified_encode,
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
        with open(file_path, "r", encoding="UTF-8") as file:
            for line in file:
                imgPath = os.path.join(self.data_dir, line.replace("\n", ""))
                depthPath = os.path.join(
                    self.data_dir,
                    imgPath.replace("/image/", "/depth/")
                    .replace("_final_hdf5", "_geometry_hdf5")
                    .replace("color.npy", "depth_meters.npy"),
                )
                semPath = os.path.join(
                    self.data_dir,
                    imgPath.replace("/image/", "/semantic/")
                    .replace("_final_hdf5", "_geometry_hdf5")
                    .replace("color.npy", "semantic.npy"),
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
        # for image_path in self.image_paths:
        #   with h5py.File(image_path, 'r') as image:
        #        image_np = np.array(image['dataset'])
        #        if np.isinf(image_np).any():
        #            print(image_path)

        self.length = self.paths.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):  # pylint:disable=too-complex
        [current_image_path, current_depth_path, current_seg_path] = self.paths[idx]

        # transform image, depth, segmentation into numpy array
        image_np = np.load(current_image_path)
        depth_np = np.load(current_depth_path)
        seg_np = np.load(current_seg_path)

        original_image_tensor = transforms.ToTensor()(image_np)
        image_tensor = self.image_transform(image_np).float()
        depth_tensor = self.depth_transform(depth_np).float()
        seg_tensor = self.seg_transform(seg_np).float()
        # original_seg_tensor = seg_tensor
        original_seg_tensor = semantic_encode(
            seg_tensor, self.data_flags["parameters"]["seg_classes"]
        )

        return_dict = {
            # for input to model
            "input_image": image_tensor,
            "input_segs": seg_tensor,
            # for other uses (e.g. as loss)
            "depths": depth_tensor,
            "original_image": original_image_tensor,
            "original_seg": original_seg_tensor,
        }

        # return_types specifies the data variations we need to compute
        # data_flags specifies how the input_image should be computed

        # we first compute all needed return types

        if self.data_flags["return_types"]["border"]:
            return_dict["border"] = (
                torch.from_numpy(semantic_to_border(seg_tensor.squeeze().numpy()))
                .unsqueeze(0)
                .float()
            )

        if self.data_flags["return_types"]["simplified_onehot"]:
            seg_tensor = return_dict["simplified_onehot"] = simplified_encode(
                seg_tensor, self.data_flags["parameters"]["simplified_onehot_classes"]
            )

        # then specify input_image based on that option
        if self.data_flags["type"] == "border":
            return_dict["input_image"] = torch.cat(
                (image_tensor, return_dict["border"]), dim=0
            )
        elif self.data_flags["type"] == "simplified_onehot":
            return_dict["input_image"] = torch.cat(
                (image_tensor, return_dict["simplified_onehot"]), dim=0
            )
        elif self.data_flags["type"] == "concat":
            seg_tensor = semantic_norm(
                seg_tensor, self.data_flags["parameters"]["seg_classes"]
            )
            return_dict["input_image"] = torch.cat((image_tensor, seg_tensor), dim=0)

        return return_dict


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
    crop_transform = (transforms.CenterCrop(256),)

    def image_transform(input_):
        x = resize(input_)
        tf = transforms.Compose(
            [*base_transform, *crop_transform, transforms.Normalize(mean, std)]
        )
        return tf(x)

    def depth_transform(input_):
        x = resize(input_)
        x = np.clip(x, min_depth, max_depth)
        x = (x - min_depth) / (max_depth - min_depth)
        tf = transforms.Compose([*base_transform, *crop_transform])
        return tf(x)

    def seg_transform(input_):
        x = resize(input_)
        tf = transforms.Compose([*base_transform, *crop_transform])
        return tf(x)

    return image_transform, depth_transform, seg_transform
