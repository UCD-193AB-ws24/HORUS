import pandas as pd
import math
from tqdm import tqdm
import random
from torch.utils.data import Dataset
import numpy as np
import torch
import os
import cv2


def augment_jitter(keypoints, valid_keypoints, noise=2.5):
    """
    Apply random jitter to keypoints
    """
    T, K, _ = keypoints.shape
    device = keypoints.device
    
    # Create random tensors directly on the correct device
    keypoints[:, :, :2] += (torch.rand((T, K, 2), device=device) * noise * 2 - noise)
    keypoints[:, :, 2:] += (torch.rand((T, K, 1), device=device) * 0.008 - 0.004)

    keypoints[:, :, :2] *= (torch.rand((T, 1, 2), device=device) * 0.2 + 0.9)
    keypoints[:, :, 2:] *= (torch.rand((T, 1, 1), device=device) * 0.2 + 0.9)

    keypoints[:, :, :2] += (torch.rand((T, 1, 2), device=device) * 0.2 - 0.1)

    return keypoints, valid_keypoints


def augment_rotation_xy(keypoints, max_angle=math.radians(15), center=(0.0, 0.0)):
    """
    Apply random rotation to keypoints
    """
    T, K, _ = keypoints.shape
    device = keypoints.device

    # Generate random angles in [-max_angle, +max_angle] for each frame
    angles = (torch.rand(T, device=device) * 2 - 1) * max_angle  # shape (T,)
    cos_vals = torch.cos(angles).view(T, 1, 1)
    sin_vals = torch.sin(angles).view(T, 1, 1)

    # Create rotation matrices directly on the correct device
    rotation_matrices = torch.cat([
        torch.cat([cos_vals, -sin_vals], dim=2),
        torch.cat([sin_vals, cos_vals], dim=2)
    ], dim=1).permute(0, 2, 1)

    # Subtract rotation center
    center = torch.tensor(center, device=device, dtype=keypoints.dtype).view(1, 1, 2)
    keypoints_xy = keypoints[:, :, :2] - center

    # Apply rotation with batch matrix multiplication
    keypoints_xy_rot = torch.bmm(keypoints_xy, rotation_matrices)

    # Add center back
    keypoints[:, :, :2] = keypoints_xy_rot + center

    return keypoints


def augment_framedrops(keypoints, valid_keypoints):
    """
    Randomly drop frames to simulate occlusion
    """
    T, K = valid_keypoints.shape
    device = valid_keypoints.device

    # Create mask directly on the correct device
    mask = (torch.rand((T, K), device=device) >= 0.1)
    valid_keypoints *= mask

    # Batch operations where possible to improve GPU efficiency
    drops = torch.rand(T, 3, device=device)
    
    # Apply frame drops for specific keypoint ranges
    for t in range(T):
        if drops[t, 0] < 0.10:
            valid_keypoints[t, 0:42] = 0
        if drops[t, 1] < 0.10:
            valid_keypoints[t, 42:520] = 0
        if drops[t, 2] < 0.10:
            valid_keypoints[t, 520:553] = 0

    # Additional drops with different probabilities
    global_drops = torch.rand(2, device=device)
    if global_drops[0] < 0.10:
        valid_keypoints[:, 520:553] = 0
    if global_drops[1] < 0.5:
        valid_keypoints[:, 42:520] = 0

    return keypoints, valid_keypoints


def augment_crop(keypoints, width, height, max_shift=0.3, crop_size_variation=0.3):
    """
    Apply random cropping augmentation
    """
    device = keypoints.device
    
    crop_w = int(width * (1 - random.uniform(0, crop_size_variation)))
    crop_h = int(height * (1 - random.uniform(0, crop_size_variation)))

    max_w_shift = int(max_shift * width)
    max_h_shift = int(max_shift * height)

    start_w = width // 2 - crop_w // 2 + random.randint(-max_w_shift, max_w_shift)
    start_h = height // 2 - crop_h // 2 + random.randint(-max_h_shift, max_h_shift)

    # Create offset tensor directly on the correct device
    offset = torch.tensor([start_w, start_h, 0], device=device)
    keypoints = keypoints - offset
    
    return keypoints, crop_w, crop_h


def sample_indices(length, target_length, augment=False):
    """
    Sample indices from video frames
    """
    if length > target_length:
        if augment == True:
            start = random.randint(0, length - target_length)
            indices = torch.linspace(start, start + target_length + random.randint(-15, 15), target_length).int()
        else:
            start = length//2 - target_length//2
            indices = torch.tensor(list(range(start, start + target_length)))
            
        indices = torch.clamp(indices, 0, length - 1)
    else:
        indices = torch.linspace(0, length - 1, target_length).int()
        shift = random.randint(-10, 10)

        if augment == True:
            indices = indices + shift
        indices = torch.clamp(indices, 0, length - 1)
        indices = indices.int()
    
    return indices


def process_keypoints(keypoints, target_length, selected_keypoints, augment=False, height=480, width=640, flipped_keypoints=None, device=None):
    """
    Process keypoints for model input with optional augmentation
    Optimized for GPU performance
    """
    # Determine device (use provided device or CUDA if available)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Sample indices and move to device
    indices = sample_indices(len(keypoints), target_length, augment=augment)
    
    # Move data to designated device immediately
    keypoints = torch.tensor(keypoints[indices], device=device)

    # Compute valid keypoints mask
    valid_keypoints = keypoints != -1
    valid_keypoints = torch.all(valid_keypoints == 1, dim=-1)

    # Apply augmentations if needed
    if augment == True:
        if random.random() < 0.5 and flipped_keypoints is not None:
            # Flip keypoints horizontally
            width_tensor = torch.tensor([width], dtype=keypoints.dtype, device=device)
            keypoints[:,:,0] = -keypoints[:,:,0] + width_tensor
            selected_keypoints = flipped_keypoints
            
        # Apply augmentations in sequence
        keypoints = augment_rotation_xy(keypoints, center=(height//2, width//2))
        keypoints, width, height = augment_crop(keypoints, width, height)
        keypoints, valid_keypoints = augment_jitter(keypoints, valid_keypoints, noise=height/180)
        keypoints, valid_keypoints = augment_framedrops(keypoints, valid_keypoints)

    # Select required keypoints
    keypoints = keypoints[:, selected_keypoints, :]
    valid_keypoints = valid_keypoints[:, selected_keypoints]
    
    # Scale coordinates
    scale = torch.tensor([1/width, 1/height, 1], dtype=keypoints.dtype, device=device)
    keypoints = keypoints * scale
    
    return keypoints, valid_keypoints


class VideoDataset(Dataset):
    def __init__(self, split, keypoints_path,
                 video_length=64, 
                 selected_keypoints=list(range(553)), 
                 flipped_selected_keypoints=None, 
                 augment=True,
                 device=None):
        """
        Dataset for video keypoints
        """
        self.split = pd.read_csv(split)
        self.keypoints_path = keypoints_path
        self.video_length = video_length
        self.selected_keypoints = selected_keypoints
        self.flipped_selected_keypoints = flipped_selected_keypoints
        self.augment = augment
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def __len__(self):
        return len(self.split)

    def __getitem__(self, i):
        """
        Get a single item from the dataset
        """
        idx = self.split['idx'][i]
        width = self.split['width'][i]
        height = self.split['height'][i]
        video_name = self.split['file'][i]
        name, extension = os.path.splitext(video_name) 
        
        # Load keypoints
        keypoints_name = name + '.npz'
        keypoints = np.load(os.path.join(self.keypoints_path, keypoints_name))['keypoints']

        # Process keypoints with device awareness
        keypoints, valid_keypoints = process_keypoints(
            keypoints, 
            self.video_length, 
            self.selected_keypoints, 
            augment=self.augment,
            flipped_keypoints=self.flipped_selected_keypoints,
            height=height,
            width=width,
            device=self.device
        )
        
        return keypoints, valid_keypoints, idx