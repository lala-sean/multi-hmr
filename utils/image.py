# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import torch
import numpy as np

IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]


def normalize_rgb(img, imagenet_normalization=True):
    """
    Args:
        - img: np.array - (W,H,3) - np.uint8 - 0/255
    Return:
        - img: np.array - (3,W,H) - np.float - -3/3
    """
    img = img.astype(np.float32) / 255.
    img = np.transpose(img, (2,0,1))
    if imagenet_normalization:
        img = (img - np.asarray(IMG_NORM_MEAN).reshape(3,1,1)) / np.asarray(IMG_NORM_STD).reshape(3,1,1)
    img = img.astype(np.float32)
    return img

def denormalize_rgb(img, imagenet_normalization=True):
    """
    Args:
        - img: np.array - (3,W,H) - np.float - -3/3
    Return:
        - img: np.array - (W,H,3) - np.uint8 - 0/255
    """
    if imagenet_normalization:
        img = (img * np.asarray(IMG_NORM_STD).reshape(3,1,1)) + np.asarray(IMG_NORM_MEAN).reshape(3,1,1)
    img = np.transpose(img, (1,2,0)) * 255.
    img = img.astype(np.uint8)
    return img
def denormalize_rgb_tensor(img, imagenet_normalization=True):
    """
    Args:
        - img: torch.Tensor - (3,H,W) or (B,3,H,W) - float (normalized)
    Return:
        - img: torch.Tensor - (3,H,W) or (B,3,H,W) - uint8 [0,255]
    """
    mean = torch.tensor(IMG_NORM_MEAN, device=img.device, dtype=img.dtype)
    std = torch.tensor(IMG_NORM_STD, device=img.device, dtype=img.dtype)
    if img.dim() == 4:
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    else:
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)
    if imagenet_normalization:
        img = img * std + mean
    img = (img * 255).clamp(0, 255).to(torch.uint8)
    return img

def unpatch(data, patch_size=14, c=3, img_size=224):
    # c = 3
    if len(data.shape) == 2:
        c=1
        data = data[:,:,None].repeat([1,1,patch_size**2])

    B,N,HWC = data.shape
    HW = patch_size**2
    c = int(HWC / HW)
    h = w = int(N**.5)
    p = q = int(HW**.5)
    data = data.reshape([B,h,w,p,q,c])
    data = torch.einsum('nhwpqc->nchpwq', data)
    return data.reshape([B,c,img_size,img_size])