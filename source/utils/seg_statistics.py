import os

import numpy as np

from tqdm import tqdm

if __name__ == "__main__":
    data_location = "/cluster/project/infk/courses/252-0579-00L/group22_semanticMD/HyperSim_Data_extracted"
    file_path = "/cluster/project/infk/courses/252-0579-00L/group22_semanticMD/semantic_md/data/train_img_path_extracted.txt"
    seg_paths = []

    with open(file_path, "r", encoding="UTF-8") as file:
        for line in file:
            imgPath = os.path.join(data_location, line.replace("\n", ""))

            semPath = os.path.join(
                data_location,
                imgPath.replace("/image/", "/semantic/")
                .replace("_final_hdf5", "_geometry_hdf5")
                .replace("color.npy", "semantic.npy"),
            )

            if os.path.exists(imgPath) and os.path.exists(semPath):
                seg_paths.append(semPath)

    n = 42
    class_counts = np.zeros(42)
    for path in tqdm(seg_paths):
        seg = np.load(path).astype(int)
        classes, counts = np.unique(seg, return_counts=True)
        class_counts[classes] += counts
    print((class_counts * 100 / class_counts.sum()))
    print(np.argsort(class_counts)[::-1][:n])
