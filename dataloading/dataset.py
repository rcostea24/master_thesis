import nibabel as nib
import numpy as np
import cv2
import os
import scipy.ndimage
import torch
from torch.utils.data import Dataset

TARGET_SPACE_DIM = [100, 100, 70]
TARGET_TIME_DIM = 140

class NiftiDataset(Dataset):
    def __init__(self, csv_file):
        nifti_files = self.get_nifti_files(data_root)
        self.data = {"scans": [], "labels": []}

        for nifti_file in nifti_files:
            nib_obj = nib.load(nifti_file)
            img = nib_obj.get_fdata()

            if len(img.shape) < 4:
                img = np.expand_dims(img, axis=3)

            zoom_factor = (TARGET_SPACE_DIM[0] / img.shape[0], TARGET_SPACE_DIM[1] / img.shape[1], TARGET_SPACE_DIM[2] / img.shape[2], 1)

            crt_space_dim = list(img.shape[:3])
            crt_time_dim = img.shape[-1]

            if crt_time_dim >= TARGET_TIME_DIM:
                img = img[:, :, :, :TARGET_TIME_DIM]
            else:
                padded_img = np.zeros(shape=(crt_space_dim + [TARGET_TIME_DIM - crt_time_dim]))
                img = np.concatenate([img, padded_img], axis=3)

            img = scipy.ndimage.zoom(img, zoom_factor)
            self.data["scans"].append(img)

            absolute_path = nifti_file.replace(data_root, "")
            subject_id = absolute_path.split(os.sep)[0]

            

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
