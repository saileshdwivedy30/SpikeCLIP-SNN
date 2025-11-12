import cv2
import torch
import numpy as np
import imageio
import os
import torch.nn as nn
import random
# Save Network 
def save_network(network, save_path):
    if isinstance(network, nn.DataParallel):
        network = network.module
    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, save_path)

def set_random_seed(seed):
    """Set random seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)


def middleTFI(spike, middle, window):
    B, C, H, W = spike.shape
    device = spike.device
    lindex, rindex = torch.zeros([B,1, H, W], device=device), torch.zeros([B,1,H, W], device=device)
    l, r = middle+1, middle+1
    for r in range(middle+1, middle + window+1):
        l = l - 1
        if l>=0:
            newpos = spike[:,l:l+1, :, :]*(1 - torch.sign(lindex)) 
            distance = l*newpos
            lindex += distance
        if r<C:
            newpos = spike[:,r:r+1, :, :]*(1 - torch.sign(rindex))
            distance = r*newpos
            rindex += distance
    rindex[rindex==0] = window+middle
    lindex[lindex==0] = middle-window
    interval = rindex - lindex
    tfi = 1.0 / interval
    return tfi

def center_crop_224(arr):
    """
    Center crop array to 224x224.
    arr: [T, H, W] or [B, T, H, W] numpy array or torch tensor
    Returns: cropped array with spatial dimensions 224x224
    """
    if len(arr.shape) == 3:
        # [T, H, W]
        T, H, W = arr.shape
        th, tw = 224, 224
        top = max(0, (H - th) // 2)
        left = max(0, (W - tw) // 2)
        return arr[:, top:top+th, left:left+tw]
    elif len(arr.shape) == 4:
        # [B, T, H, W]
        B, T, H, W = arr.shape
        th, tw = 224, 224
        top = max(0, (H - th) // 2)
        left = max(0, (W - tw) // 2)
        return arr[:, :, top:top+th, left:left+tw]
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")

def make_voxel_224(spike, bin_size=4, target_bins=50):
    """
    Convert spike sequence to voxel representation with dynamic temporal binning.
    
    Args:
        spike: [B, T, 224, 224] float32 tensor
        bin_size: Number of frames per bin (default: 4)
        target_bins: Target number of bins (default: 50 for LRN network)
    
    Returns:
        voxel: [B, target_bins, 224, 224] tensor
    """
    import torch.nn.functional as F
    B, T, H, W = spike.shape
    
    # Truncate to multiple of bin_size
    usable_T = (T // bin_size) * bin_size
    if usable_T < T:
        spike = spike[:, :usable_T]  # truncate tail
        T = usable_T
    
    num_bins = T // bin_size
    
    # Reshape and sum along bin_size dimension
    # [B, T, H, W] -> [B, num_bins, bin_size, H, W] -> [B, num_bins, H, W]
    voxel = spike.reshape(B, num_bins, bin_size, H, W).sum(dim=2)  # [B, num_bins, H, W]
    
    # Resample along time to target_bins if needed
    if num_bins != target_bins:
        # Linear interpolate on time axis using trilinear interpolation
        # voxel: [B, num_bins, H, W] -> treat time as channel dimension for interpolation
        voxel = voxel.unsqueeze(1)  # [B, 1, num_bins, H, W]
        voxel = F.interpolate(voxel, size=(target_bins, H, W), mode='trilinear', align_corners=False)
        voxel = voxel.squeeze(1)  # [B, target_bins, H, W]
    
    return voxel

def save_opt(opt,opt_path):
    with open(opt_path, 'w') as f:
        for key, value in vars(opt).items():
            f.write(f"{key}: {value}\n")

def normal_img(img,RGB = True,nor = True):
    if nor:
        img = 255 * ((img - img.min()) / (img.max() - img.min()))
    if (img.shape[0] == 3 or img.shape[0] == 1) and isinstance(img,torch.Tensor):
        img = img.permute(1,2,0)
    if isinstance(img,torch.Tensor):
        img = np.array(img.detach().cpu())
    if len(img.shape) == 2:
        img = img[...,None]
    if img.shape[-1] == 1:
        img = np.repeat(img,3,axis = -1)
    img = img.astype(np.uint8)
    if RGB == False:
        img = img[...,::-1]
    return img

def save_img(path = 'test.png',img = None,nor = True):
    if nor:
        img = 255 * ((img - img.min()) / (img.max() - img.min()))
    if isinstance(img,torch.Tensor):
        img = np.array(img.detach().cpu())
    img = img.astype(np.uint8)
    cv2.imwrite(path,img)

def make_folder(path):
    os.makedirs(path,exist_ok = True)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.sum += val * n
        self.avg = self.sum / self.count

import logging
# log info
def setup_logging(log_file):
    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_file, mode='w')  # 使用'w'模式打开文件
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())
    
    
def generate_labels(file_name):
    num_part = file_name.split('/')[-1]
    non_num_part = file_name.replace(num_part, '')
    num = int(num_part)
    labels = [non_num_part + str(num + 2 * i).zfill(len(num_part)) + '.png' for i in range(-3, 4)]
    return labels
