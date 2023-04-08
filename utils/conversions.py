import numpy as np
import cv2

def semantic_to_border(seg):
    kernel = np.ones((3,3), np.uint8)
    erosion = cv2.erode(seg, kernel, iterations = 1)
    dilation = cv2.dilate(seg, kernel, iterations = 1)
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
        [64, 128, 128]
    ]

    seg = seg.astype(np.int)
    seg_img = np.zeros((seg.shape[0], seg.shape[1], 3))
    for i in range(np.max(seg)):
        seg_img[seg == i, :] = color_map[i]
    
    return seg_img