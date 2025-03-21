from matplotlib import pyplot as plt
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

def load_data(cfg, data_root_path):
    data_path = os.path.join(data_root_path, cfg["data_path"])
    
    train_dataset = NiftiDataset(
        data_path,
        "train",
        cfg["labels_mapping"],
        cfg["exp_id"]
    )

    val_dataset = NiftiDataset(
        data_path,
        "val",
        cfg["labels_mapping"],
        cfg["exp_id"]
    )

    train_dataloader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False) 

    return train_dataloader, val_dataloader

class NiftiDataset(Dataset):
    def __init__(self, data_path, split, label_mapping, exp_id):
        csv_data = pd.read_csv(os.path.join(data_path, f"{split}_annot.csv"))
        classes_to_remove = [k for k,v in label_mapping.items() if v == -1]
        csv_data = csv_data[~csv_data["label"].isin(classes_to_remove)]
        
        if not os.path.exists("figures"):
            os.makedirs("figures", exist_ok=True)

        self.data = {"scans": [], "labels": []}
        
        image_paths = csv_data["img_path"].to_list()
        labels = csv_data["label"].to_list()
        image_label_pairs = list(zip(image_paths, labels))
        desc = f"Load {split} data"
        for nifti_file, label in tqdm(image_label_pairs, total=len(image_label_pairs), desc=desc):
            nifit_img_name = nifti_file.split("\\")[-1]
            nib_obj = nib.load(os.path.join(data_path, f"{split}_img", nifit_img_name))
            img = nib_obj.get_fdata().astype(np.float32)
            self.data["scans"].append(img)
            self.data["labels"].append(int(label_mapping[label]))
            
        plt.figure(figsize=(8, 6))
        plt.hist(self.data["labels"])
        plt.xlabel("Classes")
        plt.ylabel("Counts")
        plt.title("Distribution of data")
        plt.savefig(f"figures/data_distribution_{split}_{exp_id}.jpg")
        plt.close()

    def __len__(self):
        return len(self.data["scans"])

    def __getitem__(self, idx):
        img, label = self.data["scans"][idx], self.data["labels"][idx]
        
        img = torch.tensor(img, dtype=torch.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        label = torch.tensor(label, dtype=torch.int64)
        return img, label
        
