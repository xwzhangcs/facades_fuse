from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from PIL import Image
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class MyDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frames = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frames)

    def __getitem__(self, idx):
        img_name_1 = os.path.join(self.root_dir,
                                self.data_frames.iloc[idx, 0])
        image_1 = Image.open(img_name_1)

        img_name_2 = os.path.join(self.root_dir,
                                  self.data_frames.iloc[idx, 1])
        image_2 = Image.open(img_name_2)

        img_name_gt = os.path.join(self.root_dir,
                                  self.data_frames.iloc[idx, 2])
        image_gt = Image.open(img_name_gt)

        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
            image_gt = self.transform(image_gt)
        sample = {'input_1': image_1, 'input_2': image_2, 'input_gt': image_gt}
        return sample


