import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import time
import os

class DrowsyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)

    def __len__(self):
        length = 0
        for c in self.classes:
            class_dir = os.path.join(self.root_dir, c)
            length += len(os.listdir(class_dir))
        return length

    def __getitem__(self, idx):
        

        img_label = None
        class_idx = 0

        for c in self.classes:
            class_dir = os.path.join(self.root_dir, c)
            if idx < len(os.listdir(class_dir)):
                img_name = os.listdir(class_dir)[idx]
                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_label = c
                break
            else:
                idx -= len(os.listdir(class_dir))
                class_idx += 1

        # Convert image to PIL format
        pil_img = Image.fromarray(img)

        if self.transform:
            pil_img = self.transform(pil_img)

        
        if img_label in ["yawn", "no_yawn"]:
        
            # Crop the image
            cropped_image = img[0:400, 120:550]

            # Convert the image to PIL format and apply the transform
            pil_img = transforms.ToPILImage()(cropped_image)
            
            if self.transform:
                pil_img = self.transform(pil_img)
            else:
                
                pil_img = None
                class_idx = None

        return pil_img, class_idx

