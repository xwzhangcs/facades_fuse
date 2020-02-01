# imports
import matplotlib.pyplot as plt
import numpy as np
import os

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from custom_dataset import MyDataset
from custom_autoencoder import AutoEncoder

num_epochs = 100
batch_size = 12
learning_rate = 1e-3
latent_size = 64
PATH = "facade_ae.pth"

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor()
    ]),
    'test': transforms.Compose([
        transforms.ToTensor()
    ])
}

data_dir = 'data_ae'
image_datasets = {x: MyDataset(x + '.csv', os.path.join(data_dir, x), data_transforms[x]) for x in ['train']}
dataset_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=8)
                       for x in ['train']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
print(dataset_sizes)

# GPU mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoEncoder(in_channels=1, dec_channels=16, latent_size=latent_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/facade_experiment')

# get some random training images
dataiter = iter(dataset_loaders['train'])
sample = dataiter.next()
img_1 = sample['input_1']
img_2 = sample['input_2']
images_gt = sample['input_gt']

writer.add_graph(model, (img_1, img_2))
writer.close()