import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from model import Model
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torchmetrics
import seaborn as sns

# from models.BrainModel import BrainModel

class Trainer():
    def __init__(self, cfg, logger, train_loader, val_loader):
        self.cfg = cfg
        self.device = cfg["device"]
        self.logger = logger
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.model = None
        self.best_score = 0.0
        self.best_loss = np.inf

        self.train_losses = []
        self.train_scores = []
        self.val_losses = []
        self.val_scores = []

    def train(self):
        
        assert self.cfg["septr_params"]["num_classes"] == max(list(self.cfg["labels_mapping"].values()))+1

        self.model = Model(
            self.cfg["septr_params"]
        ).to(self.device)
        
        # self.model = Model(
        #     self.cfg["resnet_params"],
        #     self.cfg["transformer_params"]
        # ).to(self.device)
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.log(f"Total trainable params: {trainable_params}")
        
        optimizer_obj = getattr(torch.optim, self.cfg["optimizer"]["name"])
        self.optimizer = optimizer_obj(self.model.parameters(), **self.cfg["optimizer"]["parameters"])
        
        scheduler_obj = getattr(torch.optim.lr_scheduler, self.cfg["scheduler"]["name"])
        self.scheduler = scheduler_obj(self.optimizer, **self.cfg["scheduler"]["parameters"])

        loss_obj = getattr(nn, self.cfg["loss_fn"]["name"])
        if "weight" in self.cfg["loss_fn"]["parameters"]:
            weight = self.cfg["loss_fn"]["parameters"]["weight"]
            weight = torch.tensor(weight).to(self.device)
            self.cfg["loss_fn"]["parameters"]["weight"] = weight
        self.loss_fn = loss_obj(**self.cfg["loss_fn"]["parameters"])
        
        self.metrics = []
        for metric in self.cfg["metrics"]:
            metric_obj = getattr(torchmetrics, metric)
            metric_instance = metric_obj(
                task="multiclass",
                num_classes=self.cfg["septr_params"]["num_classes"],
                average="macro"
            ).to(self.device)
            self.metrics.append(metric_instance)

        if not os.path.exists("saved_models"):
            os.makedirs("saved_models", exist_ok=True)
        
        for epoch in range(1, self.cfg["epochs"]+1):
            self.logger.log(f"epoch: {epoch}")

            train_step_output = self.train_step()
            logging_string = ""
            for metric, value in train_step_output.items():
                logging_string += f"\ntrain_step_{metric}: {value}"
            self.logger.log(logging_string)
            
            val_step_output = self.val_step()
            logging_string = ""
            for metric, value in val_step_output.items():
                if metric == "matrix":
                    continue
                logging_string += f"\nval_step_{metric}: {value}"
            self.logger.log(logging_string)

            self.train_losses.append(train_step_output["loss"])
            self.train_scores.append(train_step_output["MulticlassF1Score"])
            self.val_losses.append(val_step_output["loss"])
            self.val_scores.append(val_step_output["MulticlassF1Score"])

            if val_step_output["MulticlassF1Score"] > self.best_score:
                self.best_score = val_step_output["MulticlassF1Score"]
                torch.save(self.model.state_dict(), f"saved_models/best_model_{self.cfg['exp_id']}.pt")
                self.logger.log("New model saved f1score")
                self.save_confusion_matrix(val_step_output["matrix"])
            
            torch.save(self.model.state_dict(), f"saved_models/last_model_{self.cfg['exp_id']}.pt")
            if epoch % self.cfg["scheduler"]["step"] == 0:
                self.scheduler.step()

        self.plot()

    def train_step(self):
        self.model.train()
        
        num_batches = 0
        train_step_output = {
            "loss": 0.0
        }
        
        for metric in self.metrics:
            metric.reset()
            metric_name = metric.__class__.__name__
            train_step_output[metric_name] = 0.0

        for images, labels in tqdm(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            num_batches += 1

            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            
            loss = self.loss_fn(outputs, labels)
            train_step_output["loss"] += loss.item()
            loss.backward()

            self.optimizer.step()

            predictions = torch.argmax(outputs, dim=1)

            for metric in self.metrics:
                metric.update(predictions, labels)
                    
        train_step_output["loss"] /= num_batches     
        for metric in self.metrics:
            metric_name = metric.__class__.__name__
            train_step_output[metric_name] = metric.compute().item()
                
        return train_step_output

    def val_step(self):
        self.model.eval()

        all_labels = []
        all_predictions = []
        
        num_batches = 0
        val_step_output = {
            "loss": 0.0,
            "matrix": None
        }
        
        for metric in self.metrics:
            metric.reset()
            metric_name = metric.__class__.__name__
            val_step_output[metric_name] = 0.0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                num_batches += 1

                outputs = self.model(images)

                loss = self.loss_fn(outputs, labels)
                val_step_output["loss"] += loss.item()

                predictions = torch.argmax(outputs, dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                
                for metric in self.metrics:
                    metric.update(predictions, labels)

        val_step_output["loss"] /= num_batches
        for metric in self.metrics:
            metric_name = metric.__class__.__name__
            val_step_output[metric_name] = metric.compute().item()
        val_step_output["matrix"] = confusion_matrix(all_labels, all_predictions)

        return val_step_output
    
    def save_confusion_matrix(self, matrix):
        if not os.path.exists("figures"):
            os.makedirs("figures", exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.savefig(f"figures/confusion_matrix_{self.cfg['exp_id']}.jpg")
        plt.close()
        
        self.logger.log("Confusion matrix saved")

    def plot(self):
        if not os.path.exists("figures"):
            os.makedirs("figures",  exist_ok=True)

        plt.plot(range(self.cfg["epochs"]), self.train_losses, label="train loss")
        plt.plot(range(self.cfg["epochs"]), self.val_losses, label="val loss")
        plt.legend()
        plt.title("Loss")
        plt.savefig(f"figures/loss_{self.cfg['exp_id']}.jpg")
        plt.close()
        
        plt.plot(range(self.cfg["epochs"]), self.train_scores, label="train scores")
        plt.plot(range(self.cfg["epochs"]), self.val_scores, label="val scores")
        plt.title("F1 Score")
        plt.savefig(f"figures/f1_score_{self.cfg['exp_id']}.jpg")
        plt.close()
