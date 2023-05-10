import numpy as np
import torch

import cv2

def simplified_encode(seg_tensor, num_encode):
    """
    Semantic labels are 1-40(inclusive), no 0 present, -1 means unlabeled: https://github.com/apple/ml-hypersim/issues/12
    Semantic labels: https://github.com/apple/ml-hypersim/blob/main/code/cpp/tools/scene_annotation_tool/semantic_label_descs.csv
    seg_class_order is the order of relevance of the segmentation classes
    """
    seg_class_order = [1,  2, 22,  9, 38,  5,  3, 40,  7,  6, 13,  8, 35, 20,  4, 14, 39, 18, 11, 12, 
                   10, 23, 19, 36, 15, 25, 34, 17, 24, 21, 32, 16, 27, 29, 26, 33, 30, 31, 37, 28]
    seg_tensor = torch.squeeze(seg_tensor)
    seg_encoded = torch.zeros(num_encode, seg_tensor.shape[0], seg_tensor.shape[1])
    for i in range(0, num_encode):
        seg_encoded[i] = torch.eq(seg_tensor, seg_class_order[i]).float()
    return torch.squeeze(seg_encoded)


def semantic_norm(seg_tensor, num_classes):
    """-1 first clipped to 0, 1-40 normalized by dividing by 40 to bring the entire tensor values in a 0-1 scale"""

    seg_tensor = torch.clip(seg_tensor, min=0, max=None)
    seg_tensor_norm = seg_tensor / num_classes
    return seg_tensor_norm


def semantic_to_border(seg):
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(seg, kernel, iterations=1)
    dilation = cv2.dilate(seg, kernel, iterations=1)
    diff = dilation - erosion
    mask = np.zeros(diff.shape)
    mask[diff > 0] = 1

    return mask


def semantic_to_color(seg):
    color_map = [
        [0, 0, 0],
        [0, 0, 64],
        [0, 64, 0],
        [64, 0, 0],
        [0, 64, 64],
        [64, 0, 64],
        [64, 64, 0],
        [0, 0, 128],
        [0, 128, 0],
        [128, 0, 0],
        [0, 128, 128],
        [128, 0, 128],
        [128, 128, 0],
        [0, 0, 192],
        [0, 192, 0],
        [192, 0, 0],
        [0, 192, 192],
        [192, 0, 192],
        [192, 192, 0],
        [0, 0, 224],
        [0, 224, 0],
        [224, 0, 0],
        [0, 224, 224],
        [224, 0, 224],
        [224, 224, 0],
        [128, 0, 192],
        [128, 192, 0],
        [192, 0, 128],
        [128, 192, 192],
        [192, 128, 192],
        [192, 192, 128],
        [128, 0, 224],
        [128, 224, 0],
        [224, 128, 0],
        [128, 224, 224],
        [224, 128, 224],
        [224, 224, 128],
        [64, 128, 0],
        [128, 64, 0],
        [128, 64, 128],
        [64, 128, 128],
    ]

    seg = seg.astype(np.int)
    seg_img = np.zeros((seg.shape[0], seg.shape[1], 3))
    for i in range(np.max(seg)):
        seg_img[seg == i, :] = color_map[i]

    return seg_img


def test():
    # create an example 3D tensor with seg tensor values
    seg_tensor = torch.tensor(
        [[[-1.0, 1.0, 2.0, 35.0], [1.0, 22.0, 10.0, 1.0], [15.0, 9.0, 22.0, -1.0]]]
    )

    print(seg_tensor)
    print(seg_tensor.shape)
    seg_encoded = simplified_encode(seg_tensor, 4)
    print(seg_encoded)
    print(seg_encoded.shape)


if __name__ == "__main__":
    test()
