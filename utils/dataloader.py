import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import h5py
import re

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
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

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
            label_paths = os.listdir(current_label_path)

            # Filter for images that are the color images, depth maps, and segmentation paths respectively
            # join for full path
            current_image_paths = [os.path.join(current_image_path, path) for path in pre_image_paths if
                                   re.match(r"frame\.\d{4}\.color\.hdf5", path)]
            self.image_paths += current_image_paths
            current_depth_paths = [os.path.join(current_label_path, path) for path in label_paths if
                                   re.match(r"frame\.\d{4}\.depth_meters\.hdf5", path)]
            self.depth_paths += current_depth_paths
            current_seg_paths = [os.path.join(current_label_path, path) for path in label_paths if
                                 re.match(r"frame\.\d{4}\.semantic\.hdf5", path)]
            self.seg_paths += current_seg_paths

        # assert length of all is the same
        self.length = len(self.image_paths)
        assert self.length == len(self.depth_paths) == len(self.seg_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        current_image_path = self.image_paths[idx]
        current_depth_path = self.depth_paths[idx]
        current_seg_path = self.seg_paths[idx]

        # transform image, depth, segmentation into numpy array
        with h5py.File(current_image_path, 'r') as image, h5py.File(current_depth_path, 'r') as depth, \
                h5py.File(current_seg_path, 'r') as seg:
            image_np = np.array(image["dataset"])
            depth_np = np.array(depth["dataset"])
            seg_np = np.array(seg["dataset"])

            if self.transform:
                image_np = self.transform(image_np)
                depth_np = self.transform(depth_np)
                seg_np = self.transform(seg_np)

        return image_np, depth_np, seg_np


def main():
    current_path = os.path.abspath(__file__)
    root_dir = os.path.dirname(os.path.dirname(current_path))
    dataset_root_dir = os.path.join(root_dir, 'datasets', 'hypersim', 'decompressed')

    transform = ToTensor()

    dataset = HyperSimDataset(root_dir=dataset_root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    for i, batch in enumerate(dataloader):
        images, depth, seg = batch[0], batch[1], batch[2]
        print(images.size(), depth.size(), seg.size(), sep='\n', end='\n\n')


if __name__ == '__main__':
    main()
