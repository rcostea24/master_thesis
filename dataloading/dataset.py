import nibabel as nib
import numpy as np
import pandas as pd
import cv2
import os
import scipy.ndimage
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def load_data(cfg):
    cfg_transforms = []
    for transform_name in cfg["transforms"]:
        cfg_transforms.append(getattr(T, transform_name)())

    transforms = T.Compose(cfg_transforms)

    train_dataset = NiftiDataset(
        cfg["data_path"],
        "train",
        transforms
    )

    val_dataset = NiftiDataset(
        cfg["data_path"],
        "val",                 
        transforms
    )

    train_dataloader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False) 

    return train_dataloader, val_dataloader

class NiftiDataset(Dataset):
    def __init__(self, data_path, split, transforms):
        csv_data = pd.read_csv(os.path.join(data_path, f"{split}_annot.csv"))

        self.data = {"scans": [], "labels": []}
        self.transforms = transforms

        image_label_pairs = list(zip(csv_data["img_path"].to_list(), csv_data["label"].to_list()))
        desc = f"Load {split} data"
        for nifti_file, label in tqdm(image_label_pairs, total=len(image_label_pairs), desc=desc):
            nifit_img_name = nifti_file.split(os.sep)[-1]
            nib_obj = nib.load(os.path.join(data_path, f"{split}_img", nifit_img_name))
            img = nib_obj.get_fdata()
            self.data["scans"].append(img)
            self.data["labels"].append(label)

    def __len__(self):
        return len(self.data["scans"])

    def __getitem__(self, idx):
        img, label = self.data["scans"][idx], self.data["labels"][idx]
        img = self.transforms(img)
        label = torch.tensor(label, dtype=torch.int32)
        return img, label
        
