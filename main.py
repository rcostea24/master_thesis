import json
import os
import argparse

from dataloading.dataset import load_data
from evaluation.evaluator import Evaluator
from training.logger import Logger
from training.trainer import Trainer

class ExperimentRoot():
    def __init__(self):
        self.train = "experiments"
        self.evaluate = "experiments_done"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_path", default="/home/razvan")
    parser.add_argument("--exp_id", default=None)
    parser.add_argument("--mode", default="train") # train or evaluate
    args = parser.parse_args()
    
    experiment_root = getattr(ExperimentRoot(), args.mode)

    if args.exp_id is None:
        exp_cfgs = sorted(os.listdir(experiment_root))
    else:
        exp_cfgs = [f"exp_{str(args.exp_id).zfill(3)}.json"]
        
    print(exp_cfgs)
    for cfg_file_name in exp_cfgs:
        cfg_path = os.path.join(experiment_root, cfg_file_name)
    
        with open(cfg_path, "r") as file:
            cfg = json.load(file)

        train_dataloader, val_dataloader = load_data(cfg, args.data_root_path)

        if not os.path.exists("logs"):
            os.makedirs("logs", exist_ok=True)

        logger = Logger(f"logs/log_{args.mode}_{cfg['exp_id']}.txt")

        logger.log(f"{'-'*50} Parameters {'-'*50}")
        for key, value in cfg.items():
            logger.log(f"{key}: {value}")
        logger.log(f"{'-'*50}------------{'-'*50}")

        if args.mode == "train":
            trainer = Trainer(cfg, logger, train_dataloader, val_dataloader)
            trainer.train()
        elif args.mode == "evaluate":
            for which_model in ["last", "best"]:
                evaluator = Evaluator(cfg, logger, train_dataloader, val_dataloader, which_model)
                evaluator.evaluate()
