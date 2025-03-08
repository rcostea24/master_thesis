import copy
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

LABELS_MAPPING = {
    'CN': 0, # Cognitively Normal
    'SMC': 1, # Significant Memory Concern
    'MCI': 2, # Mild Cognitive Impairment
    'LMCI': 3, # Late Mild Cognitive Impairment 
    'EMCI': 4, # Early Mild Cognitive Impairment
    'AD': 5, # Alzheimer's Dementia
}

data_root_path = r"C:\Users\razva\Master\Thesis"
output_path = r"C:\Users\razva\Master\Thesis\data\adni_preprocessed_v3"
csv_files = ["train_data.csv", "val_data.csv"]
space_dim = [32, 32, 24]
time_dim = 70

for csv_file in csv_files:
    output_df = {
        "img_path": [],
        "label": []
    }

    csv_data = pd.read_csv(os.path.join(data_root_path, csv_file))
    csv_data = csv_data.fillna("")
    csv_data = csv_data[csv_data["File Path"] != ""]
    csv_data = csv_data[csv_data["Group"] != "Patient"]

    data = {"scans": [], "labels": []}

    image_label_pairs = list(zip(csv_data["File Path"].to_list(), csv_data["Group"].to_list()))
    split = "train" if "train" in csv_file else "val"
    split_dir = os.path.join(output_path, f"{split}_img")
    os.makedirs(split_dir, exist_ok=True)
    desc = f"Load {split} data"
    idx = 0

    for nifti_file, label in tqdm(image_label_pairs, total=len(image_label_pairs), desc=desc):
        try:
            nib_obj = nib.load(os.path.join(data_root_path, nifti_file))
            nifti_name = os.path.basename(nifti_file)
            img = nib_obj.get_fdata().astype(np.int16)

            if len(img.shape) < 4:
                continue

            zoom_factor = (space_dim[0] / img.shape[0], space_dim[1] / img.shape[1], space_dim[2] / img.shape[2], 1)

            crt_space_dim = list(img.shape[:3])
            crt_time_dim = img.shape[-1]

            if crt_time_dim >= time_dim:
                img = img[:, :, :, :time_dim]
            else:
                padded_img = np.zeros(shape=(crt_space_dim + [time_dim - crt_time_dim]))
                img = np.concatenate([img, padded_img], axis=3)

            img = scipy.ndimage.zoom(img, zoom_factor)

            final_img = nib.Nifti1Image(img.astype(np.int16), nib_obj.affine)

            img_path = os.path.join(split_dir, nifti_name)
            nib.save(final_img, img_path)
            output_df["img_path"].append(nifti_name)

            label = LABELS_MAPPING[label]
            output_df["label"].append(label)
            idx += 1
        except Exception as ex:
            print(ex, nifti_file)
    
    pd.DataFrame(output_df).to_csv(os.path.join(output_path, f"{split}_annot.csv"))

        


            
