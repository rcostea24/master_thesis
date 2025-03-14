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

class Trainer():
    def __init__(self, cfg, logger, train_loader, val_loader):
        self.cfg = cfg
        self.device = cfg["device"]
        self.logger = logger
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.model = None
        self.best_score = 0.0

        self.train_losses = []
        self.train_scores = []
        self.val_losses = []
        self.val_scores = []

    def train(self):
        
        assert self.cfg["septr_num_classes"] == np.max(self.cfg["labels_mapping"].values())+1

        self.model = Model(
            self.cfg["septr_channels"],
            self.cfg["septr_input_size"],
            self.cfg["septr_num_classes"]
        ).to(self.device)
        
        optimizer_obj = getattr(torch.optim, self.cfg["optimizer"])
        self.optimizer = optimizer_obj(self.model.parameters(), lr=self.cfg["lr"])

        loss_obj = getattr(nn, self.cfg["loss_fn"])
        class_weights = torch.tensor(self.cfg["class_weights"])
        self.loss_fn = loss_obj(weights=class_weights)
        
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=self.cfg["septr_num_classes"]
        ).to(self.device)
        
        self.precision = torchmetrics.Precision(
            task="multiclass",
            num_classes=self.cfg["num_classes"]
        ).to(self.device)
        
        self.recall = torchmetrics.Recall(
            task="multiclass",
            num_classes=self.cfg["num_classes"]
        ).to(self.device)
        
        self.f1_score = torchmetrics.F1Score(
            task="multiclass",
            num_classes=self.cfg["num_classes"]
        ).to(self.device)
        
        self.balanced_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass",
            num_classes=self.cfg["num_classes"],
            average="macro"
        ).to(self.device)

        if not os.path.exists("saved_models"):
            os.makedirs("saved_models")
        
        for epoch in range(self.cfg["epochs"]):
            self.logger.log(f"epoch: {epoch+1}")

            train_step_output = self.train_step()
            logging_string = f"train_step_loss: {train_step_output["train_step_loss"]} \n "
            logging_string += f"train_step_accuracy = {train_step_output["train_step_accuracy"]} \n "
            logging_string += f"train_step_precision = {train_step_output["train_step_precision"]} \n "
            logging_string += f"train_step_recall = {train_step_output["train_step_recall"]} \n "
            logging_string += f"train_step_f1_score = {train_step_output["train_step_f1_score"]} \n "
            logging_string += f"train_step_balanced_accuracy = {train_step_output["train_step_balanced_accuracy"]} \n "
            self.logger.log(logging_string)
            
            val_step_output = self.val_step()
            logging_string = f"val_step_loss: {val_step_output["val_step_loss"]} \n "
            logging_string += f"val_step_accuracy = {val_step_output["val_step_accuracy"]} \n "
            logging_string += f"val_step_precision = {val_step_output["val_step_precision"]} \n "
            logging_string += f"val_step_recall = {val_step_output["val_step_recall"]} \n "
            logging_string += f"val_step_f1_score = {val_step_output["val_step_f1_score"]} \n "
            logging_string += f"val_step_balanced_accuracy = {val_step_output["val_step_balanced_accuracy"]} \n "
            self.logger.log(logging_string)

            self.train_losses.append(train_step_output["train_step_loss"])
            self.train_scores.append(train_step_output["train_step_f1_score"])
            self.val_losses.append(val_step_output["val_step_loss"])
            self.val_scores.append(val_step_output["val_step_f1_score"])

            if val_step_output["val_step_f1_score"] > self.best_score:
                self.best_score = val_step_output["val_step_f1_score"]
                torch.save(self.model.state_dict(), f"saved_models/best_model_{self.cfg['exp_id']}.pt")
                self.logger.log("New model saved")
                self.save_confusion_matrix(val_step_output["confusion_matrix"])
            
            torch.save(self.model.state_dict(), f"saved_models/last_model_{self.cfg['exp_id']}.pt")

        self.plot()

    def train_step(self):
        self.model.train()
        total_loss = 0.0

        for images, labels in tqdm(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            total_loss += loss.item()

            self.optimizer.step()

            predictions = torch.argmax(outputs, dim=1)

            self.accuracy.update(predictions, labels)
            self.precision.update(predictions, labels)
            self.recall.update(predictions, labels)
            self.f1_score.update(predictions, labels)
            self.balanced_accuracy.update(predictions, labels)

        train_step_loss = total_loss / len(self.train_loader)
        train_step_acc = self.accuracy.compute().item()
        train_step_precision = self.precision.compute().item()
        train_step_recall = self.recall.compute().item()
        train_step_f1_score = self.f1_score.compute().item()
        train_step_balanced_accuracy = self.balanced_accuracy.compute().item()

        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1_score.reset()
        self.balanced_accuracy.reset()
        
        train_step_output = {
            "train_step_loss": train_step_loss,
            "train_step_acc": train_step_acc,
            "train_step_precision": train_step_precision,
            "train_step_recall": train_step_recall,
            "train_step_f1_score": train_step_f1_score,
            "train_step_balanced_accuracy": train_step_balanced_accuracy,
        }

        return train_step_output

    def val_step(self):
        self.model.eval()
        total_loss = 0.0

        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)

                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                matrix = confusion_matrix(all_labels, all_predictions)
                
                self.accuracy.update(predictions, labels)
                self.precision.update(predictions, labels)
                self.recall.update(predictions, labels)
                self.f1_score.update(predictions, labels)
                self.balanced_accuracy.update(predictions, labels)

        val_step_loss = total_loss / len(self.val_loader)
        val_step_acc = self.accuracy.compute().item()
        val_step_precision = self.precision.compute().item()
        val_step_recall = self.recall.compute().item()
        val_step_f1 = self.f1_score.compute().item()
        val_step_balanced_acc = self.balanced_accuracy.compute().item()

        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1_score.reset()
        self.balanced_accuracy.reset()
        
        val_step_output = {
            "val_step_loss": val_step_loss,
            "val_step_acc": val_step_acc,
            "val_step_precision": val_step_precision,
            "val_step_recall": val_step_recall,
            "val_step_f1": val_step_f1,
            "val_step_balanced_acc": val_step_balanced_acc,
            "matrix": matrix
        }

        return val_step_output
    
    def save_confusion_matrix(self, matrix):
        if not os.path.exists("figures"):
            os.makedirs("figures")
        
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
            os.makedirs("figures")

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
