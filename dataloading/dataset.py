from matplotlib import pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import cv2
import os
import monai.transforms as mt
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def load_data(cfg, data_root_path):
    data_path = os.path.join(data_root_path, cfg["data_path"])
    dataloaders = {"train": None, "val": None}
    
    for split in ["train", "val"]:
        dataset = NiftiDataset(
            data_path,
            split,
            cfg["labels_mapping"],
            cfg["augmentations"]["spatial"]
        )
        
        shuffle = True if split == "train" else False
        dataloaders[split] = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=shuffle)
        
    return dataloaders["train"], dataloaders["val"]

class NiftiDataset(Dataset):
    def __init__(self, data_path, split, label_mapping, spatial_transforms):
        csv_data = pd.read_csv(os.path.join(data_path, f"{split}_annot.csv"))
        classes_to_remove = [k for k,v in label_mapping.items() if v == -1]
        csv_data = csv_data[~csv_data["label"].isin(classes_to_remove)]
        
        if not os.path.exists("figures"):
            os.makedirs("figures", exist_ok=True)

        self.data = {"scans": [], "labels": []}
        
        self.spatial_transforms = None
        if split == "train":
            transforms_list = []
            for transform, params in spatial_transforms.items():
                tio_obj = getattr(mt, transform)
                transforms_list.append(tio_obj(**params))
            self.spatial_transforms = mt.Compose(transforms_list)
        
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

    def __len__(self):
        return len(self.data["scans"])

    def __getitem__(self, idx):
        scan, label = self.data["scans"][idx], self.data["labels"][idx]
        
        scan = torch.tensor(scan, dtype=torch.float32)
        scan = (scan - scan.mean()) / (scan.std() + 1e-8)
        scan = torch.permute(scan, (3, 0, 1, 2))
        
        if self.spatial_transforms is not None:
            scan = torch.stack([
                    self.spatial_transforms(frame)
                    for frame in scan
                ]).squeeze(1)
        
        label = torch.tensor(label, dtype=torch.int64)
        return scan, label
        
