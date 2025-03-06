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

csv_file = r"C:\Users\razva\Master1\Thesis\adni_data.csv"
csv_data = pd.read_csv(csv_file)
csv_data = csv_data.fillna("")
csv_data = csv_data[csv_data["File Path"] != ""]

val_size = int(0.20 * len(csv_data))

train_csv_data, val_csv_data = train_test_split(csv_data, test_size=val_size)
print(len(csv_data))
print(len(train_csv_data))
print(len(val_csv_data))

# train_csv_data.to_csv(r"C:\Users\razva\Master1\Thesis\train_data.csv")
# val_csv_data.to_csv(r"C:\Users\razva\Master1\Thesis\val_data.csv")

train_csv_data