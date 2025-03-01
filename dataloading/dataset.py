import nibabel as nib
import numpy as np
import pandas as pd
import cv2
import os
import scipy.ndimage
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader

def load_data(cfg):
    cfg_transforms = []
    for transform_name in cfg["transforms"]:
        cfg_transforms.append(getattr(T, transform_name)())

    transforms = T.Compose(cfg_transforms)

    train_dataset = NiftiDataset(
        cfg["train_csv_file"],
        cfg["space_dim"],
        cfg["time_dim"],
        transforms
    )

    val_dataset = NiftiDataset(
        cfg["val_csv_file"],                 
        cfg["space_dim"],
        cfg["time_dim"],
        transforms
    )

    train_dataloader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False) 

    return train_dataloader, val_dataloader

class NiftiDataset(Dataset):
    def __init__(self, csv_file, space_dim, time_dim, transforms):
        csv_data = pd.read_csv(csv_file)
        csv_data = csv_data.fillna("")
        csv_data = csv_data[csv_data["File Path"] != ""]

        self.data = {"scans": [], "labels": []}
        self.other_data = {
            "gender": csv_data["Sex"].to_list(),
            "age": csv_data["Age"].to_list()
        }
        self.transforms = transforms

        image_label_pairs = zip(csv_data["File Path"].to_list(), csv_data["Group"].to_list())
        for nifti_file, label in image_label_pairs:
            nib_obj = nib.load(nifti_file)
            img = nib_obj.get_fdata()

            if len(img.shape) < 4:
                img = np.expand_dims(img, axis=3)

            zoom_factor = (space_dim[0] / img.shape[0], space_dim[1] / img.shape[1], space_dim[2] / img.shape[2], 1)

            crt_space_dim = list(img.shape[:3])
            crt_time_dim = img.shape[-1]

            if crt_time_dim >= time_dim:
                img = img[:, :, :, :time_dim]
            else:
                padded_img = np.zeros(shape=(crt_space_dim + [time_dim - crt_time_dim]))
                img = np.concatenate([img, padded_img], axis=3)

            img = scipy.ndimage.zoom(img, zoom_factor)
            self.data["scans"].append(img)
            self.data["labels"].append(label)

    def __len__(self):
        return len(self.data["scans"])

    def __getitem__(self, idx):
        img, label = self.data["scans"][idx], self.data["labels"][idx]
        img = self.transforms(img)
        label = torch.tensor(label, dtype=torch.int32)
        return img, label
        
