#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#  MLP_Classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from header_Classifiers import set_seed
from torchmetrics.classification import MulticlassConfusionMatrix

class MLPClassifier(L.LightningModule):
    def __init__(self, class_names):
        super().__init__()
        self.fc1 = nn.Linear(52 * 13, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 6)
        self.class_names = class_names

    def forward(self, x):
        x = x.view(-1, 52 * 13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return nn.Softmax(dim=1)(x)

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
        ax.set_title("Confusion Matrix for FCNN Classifier") #Titel eingefügt
        ax.set_xlabel("Predicted Cube")  # Beschriftung der x-Achse
        ax.set_ylabel("True Cube")  # Beschriftung der y-Achse

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
