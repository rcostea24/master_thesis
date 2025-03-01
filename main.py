import argparse
import json
import os
import torch
import numpy as np
import torch.nn as nn
import nibabel as nib

from dataloading.dataset import NiftiDataset, load_data
from torch.utils.data import DataLoader
from model import Model
from septr.septr import SeparableTr
from conv_network_3d.conv_net import ConvNetwork
from training.logger import Logger
from training.trainer import Trainer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SIZE = (6, 64)
CHANNELS = 140
NUM_CLASSES = 6
BATCH_SIZE = 32

TARGET_SPACE_DIM = [64, 64, 48]
TARGET_TIME_DIM = 140

EXPERIMENTS_ROOT = r"experiments"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_path", default="/kaggle/input/image-sentence-pair-v2")
    args = parser.parse_args()

    exp_cfgs = sorted(os.listdir(EXPERIMENTS_ROOT))
    print(exp_cfgs)
    for cfg_file_name in exp_cfgs:
        cfg_path = os.path.join(EXPERIMENTS_ROOT, cfg_file_name)
    
        with open(cfg_path, "r") as file:
            cfg = json.load(file)

        train_dataloader, val_dataloader = load_data(cfg)

        if not os.path.exists("logs"):
            os.makedirs("logs")

        logger = Logger(f"logs/log_{cfg['exp_id']}.txt")

        logger.log(f"{'-'*50} Parameters {'-'*50}")
        for key, value in cfg.items():
            logger.log(f"{key}: {value}")
        logger.log(f"{'-'*50}------------{'-'*50}")

        trainer = Trainer(cfg, logger, train_dataloader, val_dataloader)

        trainer.train()

        trainer.test_step()
