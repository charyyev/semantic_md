import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, transforms
import h5py
import re
import random
from matplotlib import pyplot as plt
import time

from utils.config import args_and_config

'''
hypersim
├── decompressed
│   ├── ai_001_001                                  <--- folder which corresponds to downloaded zip file
│   │   ├── images
│   │   │   ├── scene_cam_00_final_hdf5             <--- this is where the images are stored
│   │   │   │   └── frame.0000.color.hdf5           <--- example image
│   │   │   ├── scene_cam_00_geometry_hdf5          <--- this is where the images are stored
│   │   │   │   ├── frame.0000.depth_meters.hdf5    <--- example depth map
│   │   │   │   └── frame.0000.semantic.hdf5        <--- example segmentation map
│   │   └── _detail
│   │
│   │
│   ├── ai_001_002                                  <--- second zip file, etc.
│   │   ├── images
│   │   │   ├── scene_cam_00_final_hdf5
│   │   │   │   └── frame.0000.color.hdf5
│   │   │   ├── scene_cam_00_geometry_hdf5
│   │   │   │   ├── frame.0000.depth_meters.hdf5
│   │   │   │   └── frame.0000.semantic.hdf5
│   │   └── _detail
│   │
│   │
│   ├── ...
'''


class HyperSimDataset(Dataset):
    def __init__(self, root_dir, train=True, test_split=.8, image_transform=None, depth_transform=None,
                 seg_transform=None, data_flags=None):
        '''
        Dataset class for HyperSim
        :param root_dir: the root directory of the dataset, which contains the uncompressed data
        :param train: train or test set
        :param test_split: percentage of dataset to use as test set [0, 1]
        :param transform: transform functions
        :param data_flags: "concat" to additionally get image+seg concatenated, "onehot" for one-hot encoding of seg
        '''
        self.random_seed = 0
        self.root_dir = root_dir
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
            self.data_flags = dict()

        random.seed(self.random_seed)
        np.random.seed()
        '''
        We start in the decompressed folder. First we loop over all "ai_XXX_XXX/" folders.
        Afterwards we look in the images/ folder.
        (1) The RGB images are in scene_cam_XX_final_hdf5/frame.XXXX.color.hdf5
        (2) The depth maps are in scene_cam_XX_geometry_hfd5/frame.XXXX.depth_meters.hdf5
        (3) The segmentation maps are in scene_cam_XX_geometry_hfd5/frame.XXXX.semantic.hdf5
        For all images in this folder, we collect RGB, depth, and segmentation, concatenate it to
        (image/depth/seg)_paths
        '''
        uncompressed_items = os.listdir(self.root_dir)
        uncompressed_folders = [item for item in uncompressed_items if os.path.isdir(os.path.join(self.root_dir, item))]
        uncompressed_folders = [folder for folder in uncompressed_folders if re.match(r'ai_\d{3}_\d{3}', folder)]

        self.image_paths = []
        self.depth_paths = []
        self.seg_paths = []
        for folder in uncompressed_folders:
            # In the current ai_XXX_XXX/images/ folder, look (1) for the folder containing the images, and (2) the
            # folder containing the geometry information (depth and seg)
            abs_path = os.path.join(self.root_dir, folder, 'images')
            subfolders = os.listdir(abs_path)
            subimage_folder = [folder for folder in subfolders if re.match(r'scene_cam_\d{2}_final_hdf5', folder)][0]
            sublabel_folder = [folder for folder in subfolders if re.match(r'scene_cam_\d{2}_geometry_hdf5', folder)][0]
            current_image_path = os.path.join(abs_path, subimage_folder)
            current_label_path = os.path.join(abs_path, sublabel_folder)

            # get all file names in the respective image / geometry folders
            pre_image_paths = os.listdir(current_image_path)

            # Filter for images that are the color images, depth maps, and segmentation paths respectively
            # join for full path
            image_paths = []
            for path in pre_image_paths:
                match = re.search(r"(frame\.\d{4})\.color\.hdf5", path)
                if match:
                    image_paths.append(match.group(1))

            current_image_paths = [os.path.join(current_image_path, path + '.color.hdf5') for path in image_paths]
            self.image_paths += current_image_paths
            current_depth_paths = [os.path.join(current_label_path, path + '.depth_meters.hdf5') for path in
                                   image_paths]
            self.depth_paths += current_depth_paths
            current_seg_paths = [os.path.join(current_label_path, path + '.semantic.hdf5') for path in image_paths]
            self.seg_paths += current_seg_paths

        # assert length of all is the same
        assert len(self.image_paths) == len(self.depth_paths) == len(self.seg_paths)

        # shuffle all arrays (determined by random seed), relative order stays the same
        self.image_paths = np.array(self.image_paths)
        self.depth_paths = np.array(self.depth_paths)
        self.seg_paths = np.array(self.seg_paths)
        self.paths = np.column_stack((self.image_paths, self.depth_paths, self.seg_paths))

        # split by train and test split
        split_index = int(test_split * self.paths.shape[0])
        if train:
            self.paths = self.paths[:split_index]
        else:
            self.paths = self.paths[split_index:]

        # Print images with infinity values,
        # for image_path in self.image_paths:
        #     with h5py.File(image_path, 'r') as image:
        #         image_np = np.array(image['dataset'])
        #         if np.isinf(image_np).any():
        #             print(image_path)
        #         else:
        #             print(f'None in {image_path}')

        self.length = self.paths.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        [current_image_path, current_depth_path, current_seg_path] = self.paths[idx]

        # transform image, depth, segmentation into numpy array
        with h5py.File(current_image_path, 'r') as image, h5py.File(current_depth_path, 'r') as depth, \
                h5py.File(current_seg_path, 'r') as seg:
            image_np = np.asarray(image["dataset"], dtype=np.float32)
            depth_np = self._extract_depth(depth)
            seg_np = np.asarray(seg["dataset"], dtype=np.float32)

        # plt.imshow(image_np)
        # plt.show()
        if np.isnan(image_np).any():
            print("Image contains NaN!")

        image_tensor = self.image_transform(image_np).float()
        depth_tensor = self.depth_transform(depth_np).float()
        seg_tensor = self.seg_transform(seg_np).float()

        if self.data_flags.get("onehot", False):
            nr_classes = self.data_flags["seg_classes"]
            identity_matrix = torch.eye(nr_classes).to(seg_tensor.device)
            seg_tensor = identity_matrix[seg_tensor.reshape(-1) - 1].reshape(seg_tensor.shape + (nr_classes,))
        else:
            seg_tensor = seg_tensor.unsqueeze(-1)

        if self.data_flags.get("concat", False):
            data_tensor = torch.cat((image_tensor, seg_tensor), dim=-1)
        else:
            data_tensor = image_tensor.clone()

        return {"data": data_tensor, "image": image_tensor, "depths": depth_tensor, "segs": seg_tensor}

    def _extract_image(self, image):
        # conversion adapted from here:
        # https://github.com/apple/ml-hypersim/blob/main/code/python/tools/scene_generate_images_tonemap.py
        rgb_color = np.asarray(image["dataset"], dtype=np.float32)
        render_entity_id = np.asarray(image["dataset"], dtype=np.int32)
        # assert (render_entity_id != 0).all()

        gamma = 1.0 / 2.2  # standard gamma correction exponent
        inv_gamma = 1.0 / gamma
        percentile = 90  # we want this percentile brightness value in the unmodified image...
        brightness_nth_percentile_desired = 0.8  # ...to be this bright after scaling

        valid_mask = render_entity_id != -1

        if np.all(valid_mask):
            scale = 1.0  # if there are no valid pixels, then set scale to 1.0
        else:
            brightness = 0.3 * rgb_color[:, :, 0] + 0.59 * rgb_color[:, :, 1] + 0.11 * rgb_color[:, :,
                                                                                       2]  # "CCIR601 YIQ" method for computing brightness
            brightness_valid = brightness[valid_mask]

            eps = 0.0001  # if the kth percentile brightness value in the unmodified image is less than this, set the scale to 0.0 to avoid divide-by-zero
            brightness_nth_percentile_current = np.percentile(brightness_valid, percentile)

            if brightness_nth_percentile_current < eps:
                scale = 0.0
            else:

                # Snavely uses the following expression in the code at https://github.com/snavely/pbrs_tonemapper/blob/master/tonemap_rgbe.py:
                # scale = np.exp(np.log(brightness_nth_percentile_desired)*inv_gamma - np.log(brightness_nth_percentile_current))
                #
                # Our expression below is equivalent, but is more intuitive, because it follows more directly from the expression:
                # (scale*brightness_nth_percentile_current)^gamma = brightness_nth_percentile_desired

                scale = np.power(brightness_nth_percentile_desired, inv_gamma) / brightness_nth_percentile_current

        rgb_color_tm = np.power(np.maximum(scale * rgb_color, 0), gamma)
        rgb_color_tm = np.clip(rgb_color_tm, 0, 1)
        return rgb_color_tm

    def _extract_depth(self, depth):
        depth_np = np.asarray(depth["dataset"], dtype=np.float32)
        depth_np = np.clip(depth_np, 0, 20)
        return depth_np


def depth_range():
    return 0, 5


def main():
    config = args_and_config()
    dataset_root_dir = config["data_location"]
    dataset = HyperSimDataset(root_dir=dataset_root_dir, train=True)


if __name__ == '__main__':
    main()
