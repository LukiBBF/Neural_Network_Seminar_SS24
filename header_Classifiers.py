#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# header_Classifiers.py
import sqlite3
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import random

experiment_cubes = ["orange", "white", "red", "blue", "yellow", "green"]

def number_to_string(number, objects):
    return objects[number]

def get_experiment_object_id(obj_str, objects):
    return objects.index(obj_str)

def get_sql_for_subset(subset="last_reading"):
    if subset == "last_reading":
        sql_str = "SELECT thumb_position,finger_position,thumb_current,finger_current,thumb_touch_x, thumb_touch_y , \
                  thumb_touch_z , index_touch_x, index_touch_y , index_touch_z , ring_touch_x, ring_touch_y , ring_touch_z ,\
                  subsample.sample_number,sample.object_name FROM sample,subsample \
                  where sample.sample_number=subsample.sample_number and subsample_iteration=51 ORDER BY subsample.sample_number "
    elif subset == "all_readings":
        sql_str = "SELECT thumb_position,finger_position,thumb_current,finger_current,thumb_touch_x, thumb_touch_y , \
                  thumb_touch_z , index_touch_x, index_touch_y , index_touch_z , ring_touch_x, ring_touch_y , ring_touch_z ,\
                  subsample.sample_number,sample.object_name FROM sample,subsample \
                  where sample.sample_number=subsample.sample_number ORDER BY subsample.sample_number "
    elif subset == "everything":
        sql_str = "SELECT * FROM sample,subsample \
                  where sample.sample_number=subsample.sample_number ORDER BY subsample.sample_number "
    return sql_str

class CubeDataset(Dataset):
    def __init__(self, db_file):
        connection = sqlite3.connect(db_file)
        sql_str = get_sql_for_subset(subset="last_reading")
        df = pd.read_sql_query(sql_str, connection, coerce_float=True)
        labels = df[["object_name"]].to_numpy().squeeze()
        class_ids, self.class_names = pd.factorize(labels)
        self.target_classes = nn.functional.one_hot(torch.tensor(class_ids)).to(torch.float32)
        sql_str = get_sql_for_subset(subset="all_readings")
        df = pd.read_sql_query(sql_str, connection, coerce_float=True)
        value_columns = [col for col in df.columns if col not in ("object_name", "sample_number")]
        sensor_readings = torch.tensor(np.array([df.loc[df["sample_number"] == sample_number, value_columns].to_numpy() for sample_number in range(1, len(self.target_classes) + 1)]))
        min_readings = df.loc[:,value_columns].min().to_numpy()
        max_readings = df.loc[:,value_columns].max().to_numpy()
        self.input_values = ((sensor_readings - min_readings) / (max_readings - min_readings)).to(torch.float32)

    def __len__(self):
        return len(self.target_classes)

    def __getitem__(self, idx):
        return self.input_values[idx], self.target_classes[idx]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_dataloaders(dataset, seed):
    set_seed(seed)
    train_data, val_data, test_data = random_split(dataset, [int(0.8 * len(dataset)), int(0.1 * len(dataset)), len(dataset) - int(0.8 * len(dataset)) - int(0.1 * len(dataset))],
                                                   generator=torch.Generator().manual_seed(seed))
    ###### debugging
    print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}, Test size: {len(test_data)}")  # Drucke die Größen
    
    ######
    
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, worker_init_fn=set_seed)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False, worker_init_fn=set_seed)
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False, worker_init_fn=set_seed)
    return train_loader, val_loader, test_loader
