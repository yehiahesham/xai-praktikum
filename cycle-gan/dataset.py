# Imports
from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

# Class Dataset for Horse/Zebra

class Dataset_HZ(Dataset):
    def __init__(self, root_z, root_h, transform=None):
        self.root_zebra = root_z
        self.root_horse = root_h
        self.transform = transform

        self.horse_imgs = os.listdir(self.root_horse)
        self.zebra_imgs = os.listdir(self.root_zebra)

        # We don't have equal size datasets for horses and zebras
        # So we take the maximum length from both sets
        self.dataset_length = max(len(self.zebra_imgs), len(self.horse_imgs))
        self.zebra_len = len(self.zebra_imgs)
        self.horse_len = len(self.horse_imgs)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, item_idx):
        zebra_img = self.zebra_imgs[item_idx % self.zebra_len]
        horse_img = self.horse_imgs[item_idx % self.horse_len]

        # Individual Img Paths
        zebra_path = os.join(self.root_zebra, zebra_img)
        horse_path = os.join(self.root_horse, horse_img)

        # Convert to PIL Images
        zebra_img = np.array(Image.open(zebra_path).convert("RGB"))
        horse_img = np.array(Image.open(horse_path).convert("RGB"))

        if self.transform:
            augments = self.transform(image=zebra_img, image0=horse_img)
            zebra_img = augments["image"]
            horse_img = augments["image0"]

        return zebra_img, horse_img