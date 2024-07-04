#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# LSTM_Classifier.py
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from header_Classifiers import set_seed
from torchmetrics.classification import MulticlassConfusionMatrix
import torch


class LSTMClassifier(L.LightningModule):
    def __init__(self, class_names, input_size=13, hidden_size=75, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, len(class_names))
        self.class_names = class_names

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), 75).to(self.device)
        c_0 = torch.zeros(1, x.size(0), 75).to(self.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc1(out[:, -1, :])
        out = F.relu(out)  # Aktivierungsfunktion nach der ersten Linear-Schicht
        out = self.fc2(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = nn.functional.cross_entropy(out, y.argmax(dim=1))  # Zielvariable in die richtige Form bringen
        acc = torch.sum(y.argmax(-1) == out.argmax(-1)).item() / len(y)
        self.log_dict({'train_loss': loss, 'train_acc': acc})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = nn.functional.cross_entropy(out, y.argmax(dim=1))  # Zielvariable in die richtige Form bringen
        val_acc = torch.sum(y.argmax(-1) == out.argmax(-1)).item() / len(y)
        self.log_dict({'val_loss': loss, 'val_acc': val_acc})

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = nn.functional.cross_entropy(out, y.argmax(dim=1))  # Zielvariable in die richtige Form bringen
        pred = out.argmax(-1)
        target = y.argmax(-1)
        test_acc = torch.sum(target == pred).item() / len(y)
        self.log_dict({'test_loss': loss, 'test_acc': test_acc})
        self.preds = torch.cat((self.preds, pred.cpu()))
        self.targets = torch.cat((self.targets, target.cpu()))

    def on_test_epoch_start(self):
        self.preds = torch.empty(0)
        self.targets = torch.empty(0)

    def on_test_epoch_end(self):
        metric = MulticlassConfusionMatrix(num_classes=6)  # erstellt Confusion Matrix
        metric.update(self.preds, self.targets)
        fig, ax = metric.plot()
        ax.set_xticklabels(self.class_names)
        ax.set_yticklabels(self.class_names)
        ax.set_title("Confusion Matrix for LSTM Classifier")  # Titel eingefügt
        ax.set_xlabel("Predicted Cube")  # Beschriftung der x-Achse
        ax.set_ylabel("True Cube")  # Beschriftung der y-Achse

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"}}
