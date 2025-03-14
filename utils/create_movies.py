import copy
import nibabel as nib
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

import pandas as pd
from sklearn.model_selection import train_test_split

def normalize_frame(frame):
    """Normalize the frame data to 0-255."""
    frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
    return frame.astype(np.uint8)

# Load the image
path = "/home/razvan/data/adni_preprocessed_v3/train_img"
dest_folder = "/home/razvan/data/movies"
files = os.listdir(path)
for file in files:
    file_dest_folder = os.path.join(dest_folder, file)
    img = nib.load(os.path.join(path, file))
    data = img.get_fdata().astype(np.int32)
    x_dim, y_dim, z_dim, time_stamps = data.shape

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
            frame = data[:, :, slice_index, temp_index]
            frame = normalize_frame(frame)

            rot_frame = np.rot90(frame)
            resized_frame = cv2.resize(rot_frame, (frame_size, frame_size))
            
            # Convert grayscale to BGR explicitly
            color_frame = cv2.cvtColor(resized_frame, cv2.COLOR_GRAY2BGR)
            
            out.write(color_frame)  # Ensure frames are written

        out.release()

print("All videos saved successfully.")