import numpy as np
import cv2
import torch

def simplified_encode(seg_tensor):
    ''' 
    Semantic labels are 1-40(inclusive), no 0 present, -1 means unlabeled: https://github.com/apple/ml-hypersim/issues/12 
    Semantic labels1 (1:Wall, 2:Floor) -> https://github.com/apple/ml-hypersim/blob/main/code/cpp/tools/scene_annotation_tool/semantic_label_descs.csv
    '''
    tensor_wall = torch.eq(seg_tensor, 1).float()
    tensor_floor = torch.eq(seg_tensor, 2).float()
    tensor_other = (torch.ne(seg_tensor, 1) & torch.ne(seg_tensor, 2)).float()
    seg_encoded = torch.stack((tensor_wall, tensor_floor, tensor_other), dim=0)
    return torch.squeeze(seg_encoded)

def simplified_encode_4(seg_tensor):
    ''' 
    Semantic labels are 1-40(inclusive), no 0 present, -1 means unlabeled: https://github.com/apple/ml-hypersim/issues/12 
    Semantic labels (1:Wall, 2:Floor, 22:ceiling) -> https://github.com/apple/ml-hypersim/blob/main/code/cpp/tools/scene_annotation_tool/semantic_label_descs.csv
    '''
    tensor_wall = torch.eq(seg_tensor, 1).float()
    tensor_floor = torch.eq(seg_tensor, 2).float()
    tensor_ceiling = torch.eq(seg_tensor, 22).float()
    tensor_other = (torch.ne(seg_tensor, 1) & torch.ne(seg_tensor, 2) & torch.ne(seg_tensor, 22)).float()
    seg_encoded = torch.stack((tensor_wall, tensor_floor, tensor_ceiling, tensor_other), dim=0)
    return torch.squeeze(seg_encoded)   

def semantic_norm(seg_tensor, num_classes):
    ''' -1 first clipped to 0, 1-40 normalized by dividing by 40 to bring the entire tensor values in a 0-1 scale '''
    
    seg_tensor = torch.clip(seg_tensor, min=0, max=None)
    seg_tensor_norm = seg_tensor / num_classes
    return seg_tensor_norm


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

def test():

    # create an example 3D tensor with seg tensor values
    seg_tensor = torch.tensor([[[-1., 1., 2., 35.],
                          [1., 20., 10., 1.],
                          [15., 2., 2., -1.]]])

    print(seg_tensor)
    print(seg_tensor.shape)
    seg_encoded = simplified_encode(seg_tensor)
    print(seg_encoded)
    print(seg_encoded.shape)



if __name__ == '__main__':
    test()
