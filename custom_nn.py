from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
plt.switch_backend('agg')
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
from torchvision.models.resnet import BasicBlock
from custom_dataset import MyDataset
from custom_autoencoder import AutoEncoder
from PIL import Image
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

num_epochs = 100
batch_size = 2
learning_rate = 1e-3
latent_size = 64
PATH = "facade_ae.pth"


if __name__ == "__main__":
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.ToTensor()
        ])
    }

    data_dir = 'data_ae'
    image_datasets = {x: MyDataset(x + '.csv', os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train']}
    dataset_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                      shuffle=True, num_workers=8)
                       for x in ['train']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
    print(dataset_sizes)
    dataiter = iter(dataset_loaders['train'])
    sample = dataiter.next()
    images_1 = sample['input_1']
    images_2 = sample['input_2']
    images_gt = sample['input_gt']
    print(images_1.shape)
    print(images_2.shape)
    print(images_gt.shape)

    # GPU mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoEncoder(in_channels=1, dec_channels=16, latent_size=latent_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    start_time = time.time()
    for epoch in range(num_epochs):
        batch_idx = 0
        for data in dataset_loaders['train']:
            img_1 = data['input_1'].to(device)
            img_2 = data['input_2'].to(device)
            images_gt = data['input_gt'].to(device)
            # ===================forward=====================
            _, output = model(img_1, img_2)
            loss = criterion(output, images_gt)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ### LOGGING
            if not batch_idx % 50:
                print('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f'
                      % (epoch + 1, num_epochs, batch_idx,
                         dataset_sizes['train'] // batch_size, loss))
            batch_idx = batch_idx + 1
        print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))

    # save model
    torch.save(model.state_dict(), PATH)





