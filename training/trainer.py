import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from model import Model
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

class Trainer():
    # trainer class
    def __init__(self, cfg, logger, train_loader, val_loader, test_loader):
        self.cfg = cfg
        self.logger = logger
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader

        self.model = None
        self.best_acc = 0.0

        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []

    def train(self):
        # main train method

        # init the model
        self.model = Model(
            self.cfg["vision_params"], 
            self.cfg["language_params"], 
            self.cfg["classifier_params"]
        ).to(self.cfg["device"])
        
        # init the optimizer
        optimizer_obj = getattr(torch.optim, self.cfg["optimizer"])
        self.optimizer = optimizer_obj(self.model.parameters(), lr=self.cfg["lr"])

        # get the threshold in case of binary classification
        self.threshold = 0.5
        if "threshold" in self.cfg:
            self.threshold = self.cfg["threshold"]


        # init the loss function
        loss_obj = getattr(nn, self.cfg["loss_fn"])
        self.loss_fn = loss_obj()

        if not os.path.exists("saved_models"):
            os.makedirs("saved_models")
        
        # iterate epochs
        # for epoch in range(self.cfg["epochs"]):
        #     self.logger.log(f"epoch: {epoch+1}")

        #     # train step
        #     train_step_loss, train_step_acc = self.train_step()
        #     self.logger.log(f"train_step_loss: {train_step_loss} | train_step_acc = {train_step_acc}")
            
        #     # validation step
        #     val_step_loss, val_step_acc = self.val_step()
        #     self.logger.log(f"val_step_loss: {val_step_loss} | val_step_acc = {val_step_acc}")

        #     # append epoch's metrics
        #     self.train_losses.append(train_step_loss)
        #     self.train_accs.append(train_step_acc)
        #     self.val_losses.append(val_step_loss)
        #     self.val_accs.append(val_step_acc)

        #     # save the best model
        #     if val_step_acc > self.best_acc:
        #         self.best_acc = val_step_acc
        #         torch.save(self.model.state_dict(), f"saved_models/best_model_{self.cfg['exp_id']}.pt")
        #         self.logger.log("New model saved")
            
        #     # save the last model
        #     torch.save(self.model.state_dict(), f"saved_models/last_model_{self.cfg['exp_id']}.pt")

        # # plot metrics
        # self.plot()

    def train_step(self):
        # train step method
        self.model.train()
        total_loss = 0.0
        correct_preds = 0.0
        total_preds = 0

        # iterate data
        for img_inputs, txt_input, labels in tqdm(self.train_loader):
            img_inputs, txt_input, labels = img_inputs.to(self.cfg["device"]), txt_input.to(self.cfg["device"]), labels.to(self.cfg["device"])

            # reshape labels if binary classification
            if str(type(self.loss_fn)) == "<class 'torch.nn.modules.loss.BCEWithLogitsLoss'>":
                labels = labels.view(labels.shape[0], 1).type(torch.float32)

            # reset gradients
            self.optimizer.zero_grad()
            
            # forward pass
            outputs = self.model(img_inputs, txt_input)

            # apply softmax if the output is from multiclass classification
            if str(type(self.loss_fn)) == "<class 'torch.nn.modules.loss.CrossEntropyLoss'>":
                outputs = torch.softmax(outputs, dim=1)

            # compute loss
            loss = self.loss_fn(outputs, labels)

            # compute gradients
            loss.backward()

            # update parameters
            self.optimizer.step()

            # add loss value to total loss
            total_loss += loss.item()
            
            # get the predictions from model's output
            if str(type(self.loss_fn)) == "<class 'torch.nn.modules.loss.CrossEntropyLoss'>":
                predictions = torch.argmax(outputs, dim=1)
            elif str(type(self.loss_fn)) == "<class 'torch.nn.modules.loss.BCEWithLogitsLoss'>":
                predictions = (torch.sigmoid(outputs) >= self.threshold).type(torch.int32)

            # compute the number of corrent predictions
            correct_preds += torch.sum(predictions == labels).item()
            total_preds += labels.shape[0]

        # compute the mean loss and accuracy
        train_step_loss = total_loss / len(self.train_loader)
        train_step_acc = correct_preds / total_preds

        return train_step_loss, train_step_acc

    def val_step(self):
        # val step methos
        self.model.eval()
        total_loss = 0.0
        correct_preds = 0.0
        total_preds = 0

        # disable gradient computation
        with torch.no_grad():
            for img_inputs, txt_input, labels in tqdm(self.val_loader):
                img_inputs, txt_input, labels = img_inputs.to(self.cfg["device"]), txt_input.to(self.cfg["device"]), labels.to(self.cfg["device"])

                # reshape labels if binary classification
                if str(type(self.loss_fn)) == "<class 'torch.nn.modules.loss.BCEWithLogitsLoss'>":
                    labels = labels.view(labels.shape[0], 1).type(torch.float32)

                # forward pass
                outputs = self.model(img_inputs, txt_input)

                # apply softmax if the output is from multiclass classification
                if str(type(self.loss_fn)) == "<class 'torch.nn.modules.loss.CrossEntropyLoss'>":
                    outputs = torch.softmax(outputs, dim=1)

                # compute loss
                loss = self.loss_fn(outputs, labels)
        
                # add loss value to total loss
                total_loss += loss.item()

                # get the predictions from model's output
                if str(type(self.loss_fn)) == "<class 'torch.nn.modules.loss.CrossEntropyLoss'>":
                    predictions = torch.argmax(outputs, dim=1)
                elif str(type(self.loss_fn)) == "<class 'torch.nn.modules.loss.BCEWithLogitsLoss'>":
                    predictions = (torch.sigmoid(outputs) >= self.threshold).type(torch.int32)

                # compute the number of corrent predictions
                correct_preds += torch.sum(predictions == labels).item()
                total_preds += labels.shape[0]

        # compute the mean loss and accuracy
        val_step_loss = total_loss / len(self.val_loader)
        val_step_acc = correct_preds / total_preds

        return val_step_loss, val_step_acc

    def test_step(self):
        # test step

        # read the sample submission
        output_df = pd.DataFrame(
            pd.read_csv(os.path.join(self.cfg["data_root_path"], "sample_submission.csv"))["id"]
        )
        predictions = []
                            
        # load the best model
        best_model_path = f"saved_models/best_model_{self.cfg['exp_id']}.pt"
        best_model = Model(
            self.cfg["vision_params"], 
            self.cfg["language_params"], 
            self.cfg["classifier_params"]
        ).to(self.cfg["device"])
        best_model.load_state_dict(torch.load(best_model_path))
        best_model.eval()

        # disable gradient computation
        with torch.no_grad():
            for img_inputs, txt_input, labels in tqdm(self.test_loader):
                img_inputs, txt_input, labels = img_inputs.to(self.cfg["device"]), txt_input.to(self.cfg["device"]), labels.to(self.cfg["device"])

                # forward pass
                outputs = best_model(img_inputs, txt_input)

                # get the predictions from model's output
                if str(type(self.loss_fn)) == "<class 'torch.nn.modules.loss.CrossEntropyLoss'>":
                    outputs = torch.softmax(outputs, dim=1)
                    crt_preds = torch.argmax(outputs, dim=1)
                elif str(type(self.loss_fn)) == "<class 'torch.nn.modules.loss.BCEWithLogitsLoss'>":
                    crt_preds = (torch.sigmoid(outputs) >= self.threshold).type(torch.int32)

                predictions.extend(list(crt_preds.cpu().numpy()))
                    
        # add prediction to output dataframe
        predictions = np.array(predictions)      
        output_df["label"] = predictions

        if not os.path.exists("submissions"):
            os.makedirs("submissions")

        # save results to disk
        output_df.to_csv(f"submissions/submission_{self.cfg['exp_id']}.csv", index=False)
        self.logger.log("Test results saved")
        self.logger.log(f"Validation accuracy with best model: {self.best_acc}")

        correct_preds = 0.0
        total_preds = 0

        all_labels = []
        all_predictions = []

       # disable gradient computation
        with torch.no_grad():
            for img_inputs, txt_input, labels in tqdm(self.val_loader):
                img_inputs, txt_input, labels = img_inputs.to(self.cfg["device"]), txt_input.to(self.cfg["device"]), labels.to(self.cfg["device"])

                if str(type(self.loss_fn)) == "<class 'torch.nn.modules.loss.BCEWithLogitsLoss'>":
                    labels = labels.view(labels.shape[0], 1).type(torch.float32)

                # forward pass
                outputs = best_model(img_inputs, txt_input)

                # apply softmax if the output is from multiclass classification
                if str(type(self.loss_fn)) == "<class 'torch.nn.modules.loss.CrossEntropyLoss'>":
                    outputs = torch.softmax(outputs, dim=1)

                # get the predictions from model's output
                if str(type(self.loss_fn)) == "<class 'torch.nn.modules.loss.CrossEntropyLoss'>":
                    predictions = torch.argmax(outputs, dim=1)
                elif str(type(self.loss_fn)) == "<class 'torch.nn.modules.loss.BCEWithLogitsLoss'>":
                    predictions = (torch.sigmoid(outputs) >= self.threshold).type(torch.int32)

                # compute the number of corrent predictions
                correct_preds += torch.sum(predictions == labels).item()
                total_preds += labels.shape[0]

                # store predictions and labels for confusion matrix
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        self.logger.log(f"Validation accuracy with best model: {correct_preds / total_preds}")

        # compute confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_predictions)

        # plot the confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=True, yticklabels=True)
        plt.title(f"Confusion Matrix Experiment {self.cfg['exp_id']}")
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.savefig(f"figures/{self.cfg['exp_id']}_cm.jpg")
        plt.close()
        

    def plot(self):
        # method for plotting
        if not os.path.exists("figures"):
            os.makedirs("figures")

        # plot train and val loss
        plt.plot(range(self.cfg["epochs"]), self.train_losses, label="train loss")
        plt.plot(range(self.cfg["epochs"]), self.val_losses, label="val loss")
        plt.legend()
        plt.title("Loss")
        plt.savefig(f"figures/{self.cfg['exp_id']}_loss.jpg")
        plt.close()
        
        # plot train and val accuracy
        plt.plot(range(self.cfg["epochs"]), self.train_accs, label="train acc")
        plt.plot(range(self.cfg["epochs"]), self.val_accs, label="val acc")
        plt.title("Accuracy")
        plt.savefig(f"figures/{self.cfg['exp_id']}_accuracy.jpg")
        plt.close()
