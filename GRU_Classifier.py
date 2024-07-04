#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# GRU_Classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from header_Classifiers import set_seed
from torchmetrics.classification import MulticlassConfusionMatrix

class GRUClassifier(L.LightningModule):
    def __init__(self, class_names, input_size=13, hidden_size=75, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 6)
        self.class_names = class_names

    def forward(self, x):
        h_0 = torch.zeros(2, x.size(0), 75).to(self.device)
        out, _ = self.gru(x, h_0)
        out = self.fc1(out[:, -1, :])
        out = F.relu(out)
        out = self.fc2(out)
        return nn.Softmax(dim=1)(out)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = nn.functional.cross_entropy(out, y)
        acc = torch.sum(y.argmax(-1) == out.argmax(-1)).item() / len(y)
        self.log_dict({'train_loss': loss, 'train_acc': acc})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = nn.functional.cross_entropy(out, y)
        val_acc = torch.sum(y.argmax(-1) == out.argmax(-1)).item() / len(y)
        self.log_dict({'val_loss': loss, 'val_acc': val_acc})

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = nn.functional.cross_entropy(out, y)
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
        metric = MulticlassConfusionMatrix(num_classes=6)
        metric.update(self.preds, self.targets)
        fig, ax = metric.plot()
        ax.set_xticklabels(self.class_names)
        ax.set_yticklabels(self.class_names)
        ax.set_title("Confusion Matrix for GRU Classifier") #Titel eingefügt
        ax.set_xlabel("Predicted Cube")  # Beschriftung der x-Achse
        ax.set_ylabel("True Cube")  # Beschriftung der y-Achse

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"}}