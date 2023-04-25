import numpy as np
import torch

from datasets.hypersim_dataset import HyperSimDataset
from utils.config import args_and_config


def cum_mean_std(dataset):
    n_images = len(dataset)
    mean_sum = torch.zeros(3)
    std_sum = torch.zeros(3)

    for i in range(n_images):
        image_data = dataset[i]['image']
        image_data = image_data.permute(1, 2, 0)  # Rearrange the dimensions to (H, W, C)
        mean_sum += torch.mean(image_data, dim=(0, 1))
        std_sum += torch.std(image_data, dim=(0, 1), unbiased=False)

    result = mean_sum.numpy(), std_sum.numpy(), n_images * np.ones(3)
    result = np.stack(result, axis=0)
    return result


def main():
    # Assuming `your_dataset` is an instance of your dataset class
    config = args_and_config()
    dataset_root_dir = config["data_location"]
    results = []

    # dataset1 = HyperSimDataset(root_dir=dataset_root_dir, train=True, transform=None, data_flags=None)
    # results.append(cum_mean_std(dataset1))

    dataset2 = HyperSimDataset(root_dir=dataset_root_dir, train=False, transform=None, data_flags=None)
    results.append(cum_mean_std(dataset2))

    results = np.stack(results, axis=0)
    cum_results = np.sum(results, axis=0)

    mean = cum_results[0] / cum_results[2]
    std = cum_results[1] / cum_results[2]

    print("Mean:", mean)
    print("Standard deviation:", std)


if __name__ == '__main__':
    main()
