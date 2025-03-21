import json
import os
import argparse

from dataloading.dataset import load_data
from training.logger import Logger
from training.trainer import Trainer

EXPERIMENTS_ROOT = r"experiments"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_path", default="/home/razvan")
    parser.add_argument("--exp_id", default=None)
    args = parser.parse_args()

    if args.exp_id is None:
        exp_cfgs = sorted(os.listdir(EXPERIMENTS_ROOT))
    else:
        exp_cfgs = [f"exp_{str(args.exp_id).zfill(3)}.json"]
        
    print(exp_cfgs)
    for cfg_file_name in exp_cfgs:
        cfg_path = os.path.join(EXPERIMENTS_ROOT, cfg_file_name)
    
        with open(cfg_path, "r") as file:
            cfg = json.load(file)

        train_dataloader, val_dataloader = load_data(cfg, args.data_root_path)

        if not os.path.exists("logs"):
            os.makedirs("logs", exist_ok=True)

        logger = Logger(f"logs/log_{cfg['exp_id']}.txt")

        logger.log(f"{'-'*50} Parameters {'-'*50}")
        for key, value in cfg.items():
            logger.log(f"{key}: {value}")
        logger.log(f"{'-'*50}------------{'-'*50}")

        trainer = Trainer(cfg, logger, train_dataloader, val_dataloader)

        trainer.train()
