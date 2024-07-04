#!/usr/bin/env python3
# -- coding: utf-8 --

"""
main_Classifiers.py
Script for training and testing multiple classifiers on the CubeDataset.
"""

import lightning as L
#from lightning.pytorch.loggers import CSVLogger
from header_Classifiers import CubeDataset, set_seed, get_dataloaders
from Simple_Classifier import SimpleClassifier
from LSTM_Classifier import LSTMClassifier
from GRU_Classifier import GRUClassifier
from MLP_Classifier import MLPClassifier
from CNN_Classifier import CNNClassifier
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

# Configuration
SEED = 42
MAX_EPOCHS = 120 
DATASET_PATH = "experiment.db"

def main():
    dataset = CubeDataset(DATASET_PATH)

    classifiers = { #the Classifiers need to be runned serparately by commenting the other ones out
        #"simple": SimpleClassifier(dataset.class_names),
        #"lstm": LSTMClassifier(dataset.class_names),
        #"gru": GRUClassifier(dataset.class_names),
        #"mlp": MLPClassifier(dataset.class_names),
        "cnn": CNNClassifier(dataset.class_names),
    }

    for name, classifier in classifiers.items():
        try:
            set_seed(SEED)
            train_loader, val_loader, test_loader = get_dataloaders(dataset, SEED)
            #logger = CSVLogger("logs", name=f"{name}_training_logs") 
            logger = TensorBoardLogger("logs", name=f"{name}_training_logs")

        
            print(f"Training {name} model...")
            trainer = L.Trainer(
                max_epochs=MAX_EPOCHS,
                logger=logger,
                deterministic=True,
                log_every_n_steps=1
            )
            trainer.fit(model=classifier, train_dataloaders=train_loader, val_dataloaders=val_loader)
            
            print(f"Testing {name} model...")
            trainer.test(model=classifier, dataloaders=test_loader)
        
        except Exception as e:
            print(f"An error occurred while training/testing the {name} model: {e}")

if __name__ == "__main__":
    main()