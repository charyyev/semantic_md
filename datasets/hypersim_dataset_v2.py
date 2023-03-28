import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, transforms
import h5py
import re
import random
from matplotlib import pyplot as plt

'''
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

'''


class HyperSimDataset(Dataset):
    def __init__(self, root_dir, train=True, file_path='', test_split=.8, transform=None, data_flags=None):
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
        self.transform = transform
        self.data_flags = data_flags

        if self.data_flags is None:
            self.data_flags = dict()

        random.seed(self.random_seed)
        np.random.seed()

        self.image_paths = []
        self.depth_paths = []
        self.seg_paths = []
        
        if file_path=='':
            print("File path not given for loading data...")
        with open(file_path, 'r') as file:
            for line in file:
                imgPath = os.path.join(self.root_dir, line.replace('\n',''))
                depthPath = os.path.join(self.root_dir, imgPath.replace('/image/','/depth/').replace('_final_hdf5', '_geometry_hdf5').replace('color.hdf5','depth_meters.hdf5'))
                semPath = os.path.join(self.root_dir, imgPath.replace('/image/','/semantic/').replace('_final_hdf5', '_geometry_hdf5').replace('color.hdf5','semantic.hdf5'))

                if os.path.exists(imgPath) and os.path.exists(depthPath) and os.path.exists(semPath):
                    self.image_paths.append(imgPath)
                    self.depth_paths.append(depthPath)
                    self.seg_paths.append(semPath)
                else:
                    print(imgPath)

        # assert length of all is the same
        assert len(self.image_paths) == len(self.depth_paths) == len(self.seg_paths)

        # shuffle all arrays (determined by random seed), relative order stays the same
        self.image_paths = np.array(self.image_paths)
        self.depth_paths = np.array(self.depth_paths)
        self.seg_paths = np.array(self.seg_paths)
        self.paths = np.column_stack((self.image_paths, self.depth_paths, self.seg_paths))

        self.length = self.paths.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        [current_image_path, current_depth_path, current_seg_path] = self.paths[idx]

        # transform image, depth, segmentation into numpy array
        with h5py.File(current_image_path, 'r') as image, h5py.File(current_depth_path, 'r') as depth, \
                h5py.File(current_seg_path, 'r') as seg:
            image_np = np.array(image["dataset"])
            depth_np = np.array(depth["dataset"])
            seg_np = np.array(seg["dataset"])

        if self.transform:
            transform = self.transform
        else:
            transform = transforms.ToTensor()

        image_tensor = transform(image_np).float()
        depth_tensor = transform(depth_np).float()
        seg_tensor = transform(seg_np).float()

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


def main():

    dataset_root_dir = "/cluster/project/infk/courses/252-0579-00L/group22_semanticMD/HyperSim_Data"
    test_file_path = "/cluster/project/infk/courses/252-0579-00L/group22_semanticMD/semantic_md/datasets/hypersim/test_imgPath.txt"

    transform = None

    dataset = HyperSimDataset(root_dir=dataset_root_dir, file_path=test_file_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    for i, batch in enumerate(dataloader):
        images, depth, seg = batch["image"], batch["depths"], batch["segs"]
        print(images.size(), depth.size(), seg.size(), sep='\n', end='\n\n')

        # Display the images, depth maps, and segmentation maps in separate subplots
        for idx in range(images.size(0)):
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(images[idx].permute(1, 2, 0))
            plt.title('Image')

            plt.subplot(1, 3, 2)
            plt.imshow(depth[idx].squeeze(), cmap='viridis')
            plt.title('Depth Map')

            plt.subplot(1, 3, 3)
            plt.imshow(seg[idx].squeeze(), cmap='tab20')
            plt.title('Segmentation Map')

            plt.show()
            exit(0)


if __name__ == '__main__':
    main()
