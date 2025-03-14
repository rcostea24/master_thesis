import copy
import nibabel as nib
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from einops.layers.torch import Rearrange
import torch
import torchvision
import scipy

import pandas as pd
from sklearn.model_selection import train_test_split

space_dim = [32, 32, 24]
time_dim = 70

def normalize_frame(frame):
    """Normalize the frame data to 0-255."""
    frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
    return frame.astype(np.uint8)

def resize(img):
    zoom_factor = (space_dim[0] / img.shape[0], space_dim[1] / img.shape[1], space_dim[2] / img.shape[2], 1)

    crt_space_dim = list(img.shape[:3])
    crt_time_dim = img.shape[-1]

    if crt_time_dim >= time_dim:
        img = img[:, :, :, :time_dim]
    else:
        padded_img = np.zeros(shape=(crt_space_dim + [time_dim - crt_time_dim]))
        img = np.concatenate([img, padded_img], axis=3)

    img = scipy.ndimage.zoom(img, zoom_factor)

    return img

# Load the image
path = r"C:\Users\razva\Master\Thesis\data\adni_preprocessed_v4\train_img"
dest_folder = r"C:\Users\razva\Master\Thesis\movies"
files = os.listdir(path)
for file in files:
    if not file.endswith("nii.gz"):
        continue

    file_dest_folder = os.path.join(dest_folder, file)
    nib_obj = nib.load(os.path.join(path, file))
    img = nib_obj.get_fdata().astype(np.int32)

    x_dim, y_dim, z_dim, time_stamps = img.shape

    frame_size = 256
    frame_size = frame_size if frame_size % 2 == 0 else frame_size - 1  

    num_cols = int(np.ceil(np.sqrt(time_stamps)))
    num_rows = int(np.ceil(time_stamps / num_cols))

    os.makedirs(file_dest_folder, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    for slice_index in range(z_dim):
        video_path = os.path.join(file_dest_folder, f"slice_{slice_index}.mp4")
        out = cv2.VideoWriter(video_path, fourcc, 10, (frame_size, frame_size), isColor=True)

        for temp_index in range(time_stamps):
            frame = img[:, :, slice_index, temp_index]
            frame = normalize_frame(frame)

            rot_frame = np.rot90(frame)
            resized_frame = cv2.resize(rot_frame, (frame_size, frame_size))
            
            # Convert grayscale to BGR explicitly
            color_frame = cv2.cvtColor(resized_frame, cv2.COLOR_GRAY2BGR)
            
            out.write(color_frame)  # Ensure frames are written

        out.release()

print("All videos saved successfully.")