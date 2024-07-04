# Simple_Classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from header_Classifiers import set_seed
from torchmetrics.classification import MulticlassConfusionMatrix

class SimpleClassifier(L.LightningModule):
    def __init__(self, class_names):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(52 * 13, 512),
            nn.ReLU(),
            nn.Linear(512, 6),
            nn.Softmax(dim=1)
        )
        self.class_names = class_names

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = nn.functional.cross_entropy(out, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = nn.functional.cross_entropy(out, y)
        val_acc = torch.sum(y.argmax(-1) == out.argmax(-1)).item() / len(y)
        self.log_dict({'val_loss': loss, 'val_acc': val_acc})

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
