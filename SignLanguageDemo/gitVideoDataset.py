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
    T, K, _ = keypoints.shape
    device = keypoints.device
    
    keypoints[:, :, :2] += (torch.rand((T, K, 2), device=device) * noise * 2 - noise)
    keypoints[:, :, 2:] += (torch.rand((T, K, 1), device=device) * 0.008 - 0.004)

    keypoints[:, :, :2] *= (torch.rand((T, 1, 2), device=device) * 0.2 + 0.9)
    keypoints[:, :, 2:] *= (torch.rand((T, 1, 1), device=device) * 0.2 + 0.9)

    keypoints[:, :, :2] += (torch.rand((T, 1, 2), device=device) * 0.2 - 0.1)

    return keypoints, valid_keypoints



def augment_rotation_xy(keypoints, max_angle=math.radians(15), center=(0.0, 0.0)):
    T, K, _ = keypoints.shape
    device = keypoints.device

    # Generate random angles in [-max_angle, +max_angle] for each frame
    angles = (torch.rand(T, device=device) * 2 - 1) * max_angle  # shape (T,)
    cos_vals = torch.cos(angles).view(T, 1, 1)
    sin_vals = torch.sin(angles).view(T, 1, 1)

    # Create (T, 2, 2) rotation matrices
    rotation_matrices = torch.cat([
        torch.cat([cos_vals, -sin_vals], dim=2),
        torch.cat([sin_vals, cos_vals], dim=2)
    ], dim=1).permute(0, 2, 1)

    # Subtract rotation center
    center = torch.tensor(center, device=device, dtype=keypoints.dtype).view(1, 1, 2)  # (1, 1, 2)
    keypoints_xy = keypoints[:, :, :2] - center  # (T, K, 2)

    # Apply rotation
    keypoints_xy_rot = torch.bmm(keypoints_xy, rotation_matrices)  # (T, K, 2)

    # Add center back
    keypoints[:, :, :2] = keypoints_xy_rot + center

    return keypoints

    

def augment_framedrops(keypoints, valid_keypoints):
    T, K = valid_keypoints.shape
    device = valid_keypoints.device

    mask = (torch.rand((T, K), device=device) >= 0.1)
    valid_keypoints *= mask

    for t in range(T):
        for start, end in [(0, 42), (42, 520), (520, 553)]:
            if torch.rand(1, device=device).item() < 0.10:
                valid_keypoints[t, start:end] = 0

    for start, end in [(520, 553)]:
        if torch.rand(1, device=device).item() < 0.10:
            valid_keypoints[:, start:end] = 0

    for start, end in [(42, 520)]:
        if torch.rand(1, device=device).item() < 0.5:
            valid_keypoints[:, start:end] = 0

    return keypoints, valid_keypoints

    

def augment_crop(keypoints, width, height, max_shift=0.3, crop_size_variation=0.3):
    crop_w = int(width * (1 - random.uniform(0, crop_size_variation)))
    crop_h = int(height * (1 - random.uniform(0, crop_size_variation)))

    max_w_shift = int(max_shift * width)
    max_h_shift = int(max_shift * height)

    start_w = width // 2 - crop_w // 2 + random.randint(-max_w_shift, max_w_shift)
    start_h = height // 2 - crop_h // 2 + random.randint(-max_h_shift, max_h_shift)
    #start_h = clamp(start_h, 0, h-crop_size_h)
    #start_w = clamp(start_w, 0, w-crop_size_w)

    keypoints = keypoints - torch.tensor([start_w, start_h, 0], device=keypoints.device)
    return keypoints, crop_w, crop_h
    
    

def sample_indices(length, target_length, augment=False):
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


def process_keypoints(keypoints, target_length, selected_keypoints, augment=False, height=480, width=640, flipped_keypoints=None):
    
    indices = sample_indices(len(keypoints), target_length, augment=augment)
    keypoints = torch.tensor(keypoints[indices])

    valid_keypoints = keypoints != -1
    valid_keypoints = torch.all(valid_keypoints == 1, dim=-1)



    if augment == True:
        if random.random() < 0.5 and flipped_keypoints != None:
            keypoints[:,:,0] = -keypoints[:,:,0] + torch.tensor([width], dtype=keypoints.dtype)
            selected_keypoints = flipped_keypoints
            
        keypoints = augment_rotation_xy(keypoints, center = (height//2, width//2))
        keypoints, width, height = augment_crop(keypoints, width, height)
        keypoints, valid_keypoints = augment_jitter(keypoints, valid_keypoints, noise=height/180)
        keypoints, valid_keypoints = augment_framedrops(keypoints, valid_keypoints)


    
    keypoints = keypoints[:, selected_keypoints, :]
    valid_keypoints = valid_keypoints[:, selected_keypoints]    
    keypoints = keypoints * torch.tensor([1/width, 1/height, 1], dtype=keypoints.dtype)
    return keypoints, valid_keypoints




class VideoDataset(Dataset):
    def __init__(self, split, keypoints_path,
                 video_length=64, 
                 selected_keypoints=list(range(553)), 
                 flipped_selected_keypoints=None, 
                 augment=True):

        self.split = pd.read_csv(split)
        self.keypoints_path = keypoints_path
        self.video_length = video_length
        self.selected_keypoints = selected_keypoints
        self.flipped_selected_keypoints = flipped_selected_keypoints
        self.augment = augment

        
    def __len__(self):
        return len(self.split)

    def __getitem__(self, i):
        
        idx = self.split['idx'][i]
        width = self.split['width'][i]
        height = self.split['height'][i]
        video_name = self.split['file'][i]
        name, extension = os.path.splitext(video_name) 
        
        keypoints_name = name + '.npz'
        keypoints = np.load(os.path.join(self.keypoints_path, keypoints_name))['keypoints']

        keypoints, valid_keypoints = process_keypoints(
            keypoints, 
            self.video_length, 
            self.selected_keypoints, 
            augment=self.augment,
            flipped_keypoints=self.flipped_selected_keypoints,
            height=height,
            width=width
        )
        
        return keypoints, valid_keypoints, idx



        