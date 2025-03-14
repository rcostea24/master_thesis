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
        self.best_acc = 0.0

        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []

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
        self.loss_fn = loss_obj()
        
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

            train_step_loss, train_step_acc = self.train_step()
            self.logger.log(f"train_step_loss: {train_step_loss} | train_step_acc = {train_step_acc}")
            
            val_step_loss, val_step_acc = self.val_step()
            self.logger.log(f"val_step_loss: {val_step_loss} | val_step_acc = {val_step_acc}")

            self.train_losses.append(train_step_loss)
            self.train_accs.append(train_step_acc)
            self.val_losses.append(val_step_loss)
            self.val_accs.append(val_step_acc)

            if val_step_acc > self.best_acc:
                self.best_acc = val_step_acc
                torch.save(self.model.state_dict(), f"saved_models/best_model_{self.cfg['exp_id']}.pt")
                self.logger.log("New model saved")
                self.save_confusion_matrix()
            
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
        train_step_f1 = self.f1_score.compute().item()
        train_step_balanced_acc = self.balanced_accuracy.compute().item()

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
            "train_step_f1": train_step_f1,
            "train_step_balanced_acc": train_step_balanced_acc,
        }

        return train_step_output

    def val_step(self):
        self.model.eval()
        total_loss = 0.0
        correct_preds = 0.0
        total_preds = 0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)

                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)
                correct_preds += torch.sum(predictions == labels).item()
                total_preds += labels.shape[0]

        val_step_loss = total_loss / len(self.val_loader)
        val_step_acc = correct_preds / total_preds

        return val_step_loss, val_step_acc
    
    def save_confusion_matrix(self):
        if not os.path.exists("figures"):
            os.makedirs("figures")
            
        best_model_path = f"saved_models/best_model_{self.cfg['exp_id']}.pt"
        self.model.load_state_dict(torch.load(best_model_path, weights_only=True))
        self.model.eval()
        
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                predictions = torch.argmax(outputs, dim=1)
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
        
        cm = confusion_matrix(all_labels, all_predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.savefig(f"figures/{self.cfg['exp_id']}_confusion_matrix.jpg")
        plt.close()
        
        self.logger.log("Confusion matrix saved")

    def plot(self):
        if not os.path.exists("figures"):
            os.makedirs("figures")

        plt.plot(range(self.cfg["epochs"]), self.train_losses, label="train loss")
        plt.plot(range(self.cfg["epochs"]), self.val_losses, label="val loss")
        plt.legend()
        plt.title("Loss")
        plt.savefig(f"figures/{self.cfg['exp_id']}_loss.jpg")
        plt.close()
        
        plt.plot(range(self.cfg["epochs"]), self.train_accs, label="train acc")
        plt.plot(range(self.cfg["epochs"]), self.val_accs, label="val acc")
        plt.title("Accuracy")
        plt.savefig(f"figures/{self.cfg['exp_id']}_accuracy.jpg")
        plt.close()
