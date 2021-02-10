import os, sys

import random
import torch
import matplotlib.pyplot as plt

import numpy  as np
import PIL
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
from torchvision import datasets
import json

class BarrettData(data.Dataset):
    def __init__(self, img_dir, batch_size=32):
        self.batch_size = batch_size
        self.img_dir = img_dir
        X, Y = "X", "Y"
        self.training_data = os.listdir(self.img_dir)
        
        self.training_data.sort()
        print("Training Data length: ", len(self.training_data))

    def __getitem__(self, index):
        wsi_idx = self.training_data[index]
        wsi_patch_tensor = torch.load(os.path.join(self.img_dir,X,wsi_idx))
        wsi_patch_labels = torch.load(os.path.join(self.img_dir,Y,wsi_idx))
        n_patches, _ = wsi_patch_tensor.shape
        batch = torch.randint(0,self.batch_sizes)
        x = wsi_patch_tensor[batch, :]
        y = wsi_patch_labels[batch]
        print(x)
        print(y)
        print(x.shape)
        print(y.shape)
        raise ValueError()
        
        return x, y
   
    def create_input_labels(self, wsi_set, label_set):
        

    def __len__(self):
        return len(self.training_data)
    