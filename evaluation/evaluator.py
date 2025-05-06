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
        self.which_model = which_model

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
            metric_obj = getattr(torchmetrics, metric)
            metric_instance = metric_obj(
                task="multiclass",
                num_classes=self.cfg["num_classes"],
                average="macro"
            ).to(self.device)
            self.metrics.append(metric_instance)
        
        for split in ["train", "val"]:
            output = self.evaluate_step(split)
            metrics_per_class = self.compute_metrics_per_class(output["matrix"])
            
            logging_string = f"{split} metrics"
            
            for metric, value in output.items():
                logging_string += f"\n{split}_{metric}: {value}"
                
            for class_name, metrics in metrics_per_class.items():
                for metric, value in metrics.items():
                    logging_string += f"\n{split}_{class_name}_{metric}: {value}"
                self.save_metrics_image(metrics, split, class_name)
            
            self.logger.log(logging_string)
            self.save_metrics_image(output, split, "all")

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
    
    def save_confusion_matrix(self, matrix, split):
        if not os.path.exists("figures_eval"):
            os.makedirs("figures_eval", exist_ok=True)

        matrix_percent = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis] * 100

        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix_percent, annot=True, fmt=".2f", cmap='Blues', cbar_kws={'label': 'Percentage (%)'})
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix {split}_{self.which_model}_{self.cfg['exp_id']}")
        plt.savefig(f"figures_eval/cm_{split}_{self.which_model}_{self.cfg['exp_id']}.jpg")
        plt.close()

        self.logger.log("Confusion matrix saved")
        
    def save_metrics_image(self, metrics, split, class_name="all"):
        if not os.path.exists("figures_eval"):
            os.makedirs("figures_eval", exist_ok=True)

        # Start the figure
        plt.figure(figsize=(6, len(metrics) * 0.6 + 1))
        plt.axis('off')  # Hide axes

        # Title
        title = f"Evaluation Metrics\n{split}_{self.which_model}\n{class_name}\n{self.cfg['exp_id']}"
        plt.title(title, fontsize=14, weight='bold', loc='left')

        # Write each metric as a line
        for i, (key, value) in enumerate(metrics.items()):
            if key == "matrix":
                self.save_confusion_matrix(value, split)
            else:
                plt.text(0.01, 1 - (i + 1) * 0.1, f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}",
                        fontsize=12, ha='left')

        # Save and close
        save_path = f"figures_eval/metrics_{split}_{self.which_model}_{class_name}_{self.cfg['exp_id']}"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        print(f"Metrics image saved")
        
    def compute_metrics_per_class(self, confusion_matrix):
        metrics = {}
        class_names = ["CN", "MCI", "AD"]

        for i in range(self.cfg["num_classes"]):
            TP = confusion_matrix[i, i]
            FP = confusion_matrix[:, i].sum() - TP
            FN = confusion_matrix[i, :].sum() - TP
            TN = confusion_matrix.sum() - TP - FP - FN

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            class_label = class_names[i]
            metrics[class_label] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }

        return metrics

