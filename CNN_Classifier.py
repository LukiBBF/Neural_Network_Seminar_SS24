#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 00:25:56 2024

@author: louiseschop
"""

# CNN_Classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from header_Classifiers import set_seed
from torchmetrics.classification import MulticlassConfusionMatrix

class CNNClassifier(L.LightningModule):
    def __init__(self, class_names):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(13, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(128 * 13, 256),  # Adjust the input size if necessary
            nn.ReLU(),
            nn.Linear(256, len(class_names)),
            nn.Softmax(dim=1)
        )
        self.class_names = class_names

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to [BatchSize, Channels, Length]
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = nn.functional.cross_entropy(out, y.argmax(dim=1))
        acc = torch.sum(y.argmax(-1) == out.argmax(-1)).item() / len(y)
        self.log_dict({'train_loss': loss, 'train_acc': acc})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = nn.functional.cross_entropy(out, y.argmax(dim=1))
        val_acc = torch.sum(y.argmax(-1) == out.argmax(-1)).item() / len(y)
        self.log_dict({'val_loss': loss, 'val_acc': val_acc})

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = nn.functional.cross_entropy(out, y.argmax(dim=1))
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
        ax.set_title("Confusion Matrix for CNN Classifier")
        ax.set_xlabel("Predicted Cube")
        ax.set_ylabel("True Cube")
        self.logger.experiment.add_figure("Confusion Matrix", fig)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"}}
