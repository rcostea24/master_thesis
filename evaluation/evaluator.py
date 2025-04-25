import importlib
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from models.SeptrModel import SeptrModel
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torchmetrics
import seaborn as sns


class Evaluator():
    def __init__(self, cfg, logger, train_loader, val_loader, which_model="best"):
        self.cfg = cfg
        self.device = cfg["device"]
        self.logger = logger
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_path = os.path.join("saved_models", f"{which_model}_model_{cfg['exp_id']}.pt")
        self.logger.log(f"{which_model} model")

        self.model = None

    def evaluate(self):
        
        assert self.cfg["num_classes"] == max(list(self.cfg["labels_mapping"].values()))+1

        model_name = self.cfg["model_name"]
        module = importlib.import_module(f"models.{model_name}")
        model_class = getattr(module, model_name)
        
        self.model = model_class(
            self.cfg[f"params_{model_name}"],
            self.cfg["num_classes"]
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(self.model_path))
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.log(f"Total trainable params: {trainable_params}")
        
        self.metrics = []
        for metric in self.cfg["metrics"]:
            for average in ["macro", "micro"]:
                metric_obj = getattr(torchmetrics, metric)
                metric_instance = metric_obj(
                    task="multiclass",
                    num_classes=self.cfg["num_classes"],
                    average=average
                ).to(self.device)
                self.metrics.append(metric_instance)
        
        for split in ["train", "val"]:
            output = self.evaluate_step(split)
            logging_string = f"{split} metrics"
            for metric, value in output.items():
                logging_string += f"\n{split}_step_{metric}: {value}"
            self.logger.log(logging_string)

    def evaluate_step(self, split):
        self.model.eval()

        all_labels = []
        all_predictions = []

        output = {
            "matrix": None
        }
        
        for metric in self.metrics:
            metric.reset()
            metric_name = metric.__class__.__name__
            output[metric_name] = 0.0

        with torch.no_grad():
            for images, labels in tqdm(getattr(self, f"{split}_loader")):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)

                predictions = torch.argmax(outputs, dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                
                for metric in self.metrics:
                    metric.update(predictions, labels)

        for metric in self.metrics:
            metric_name = metric.__class__.__name__
            output[metric_name] = metric.compute().item()
        output["matrix"] = confusion_matrix(all_labels, all_predictions)

        return output

