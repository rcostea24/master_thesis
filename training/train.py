import numpy as np
import cv2
import pandas as pd
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from model import Model

DEVICE = "cuda"

INPUT_SIZE = (6, 64)
CHANNELS = 140
NUM_CLASSES = 6
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3

def train_step(model, train_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    correct_preds = 0.0
    total_preds = 0

    for img_inputs, txt_input, labels in tqdm(train_loader):
        img_inputs, txt_input, labels = img_inputs.to(DEVICE), txt_input.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        
        outputs = model(img_inputs, txt_input)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        correct_preds += torch.sum(predictions == labels).item()
        total_preds += labels.shape[0]

    train_step_loss = total_loss / len(train_loader)
    train_step_acc = correct_preds / total_preds

    return train_step_loss, train_step_acc

def val_step(model, val_loader, loss_fn):
    model.eval()
    total_loss = 0.0
    correct_preds = 0.0
    total_preds = 0

    with torch.no_grad():
        for img_inputs, txt_input, labels in tqdm(val_loader):
            img_inputs, txt_input, labels = img_inputs.to(DEVICE), txt_input.to(DEVICE), labels.to(DEVICE)

            outputs = model(img_inputs, txt_input)
            loss = loss_fn(outputs, labels)
    
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct_preds += torch.sum(predictions == labels).item()
            total_preds += labels.shape[0]

    val_step_loss = total_loss / len(val_loader)
    val_step_acc = correct_preds / total_preds

    return val_step_loss, val_step_acc

def train(train_dataloader, val_dataloader):
    model = Model(CHANNELS, INPUT_SIZE, NUM_CLASSES)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_model = None

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(EPOCHS):
        print(f"epoch: {epoch+1}")
        train_step_loss, train_step_acc = train_step(model, train_dataloader, optimizer, loss_fn)
        print(f"train_step_loss: {train_step_loss} | train_step_acc = {train_step_acc}")
        
        val_step_loss, val_step_acc = val_step(model, val_dataloader, loss_fn)
        print(f"val_step_loss: {val_step_loss} | val_step_acc = {val_step_acc}")

        train_losses.append(train_step_loss)
        train_accs.append(train_step_acc)
        val_losses.append(val_step_loss)
        val_accs.append(val_step_acc)

        if val_step_acc > best_acc:
            best_acc = val_step_acc
            best_model = model

        torch.save(best_model.state_dict(), "best_model.pt")

    plt.plot(range(EPOCHS), train_losses, label="train loss")
    plt.plot(range(EPOCHS), val_losses, label="val loss")
    plt.legend()
    plt.title("Loss")
    plt.savefig("loss.jpg")
    plt.close()
    
    plt.plot(range(EPOCHS), train_accs, label="train acc")
    plt.plot(range(EPOCHS), val_accs, label="val acc")
    plt.title("Accuracy")
    plt.savefig("accuracy.jpg")
    plt.close()