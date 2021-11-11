import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset



class EplDataset(Dataset):
    def __init__(self,train = True, transform=None,target_transform=None):
        if train :
            self.data = pd.read_csv("./data/processed/numeric_train.csv")
        else :
            self.data = pd.read_csv("./data/processed/numeric_test.csv")
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input = self.data.iloc[idx,:-1].to_numpy(dtype=float)
        input = np.expand_dims(input, axis=1)
        label = self.data.iloc[idx,-1]
        label = np.float32(label)
        if self.transform :
            input = self.transform(input)
            input = input.float()
        if self.target_transform :
            label = self.target_transform(label)
        
        return input, label

    # def get_label_dictionary() :
