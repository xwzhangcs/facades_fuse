from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
plt.switch_backend('agg')
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
from custom_dataset import MyDataset
from custom_autoencoder import AutoEncoder
from PIL import Image
import cv2
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

batch_size = 1
PATH = "facade_ae.pth"
latent_size = 64

if __name__ == "__main__":
    # Define transforms
    transformations = transforms.Compose([transforms.ToTensor()])
    # Define custom dataset
    test_data = MyDataset("test.csv", "data_ae/test", transformations)
    # Define data loader
    test_dataset_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    # GPU mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model
    model = AutoEncoder(in_channels=1, dec_channels=16, latent_size=latent_size)
    model.load_state_dict(torch.load(PATH))
    model.eval()

    for i_batch, test_batch in enumerate(test_dataset_loader):
        img_1 = test_batch['input_1']
        img_2 = test_batch['input_2']
        images_gt = test_batch['input_gt']
        print(i_batch, images_gt.size())
        _, output = model(img_1, img_2)
        # output is resized into a batch of iages
        output = output.view(batch_size, 1, 128, 128)
        # use detach when it's an output that requires_grad
        output = output.detach().numpy()
        output = np.squeeze(output)
        output = np.where(output < 0.1, 0, 255)
        #im = Image.fromarray(output)
        #matplotlib.image.imsave('eval/facade_' + str(i_batch) + '.png', im)
        cv2.imwrite('eval_fuse/facade_' + str(i_batch) + '.png', output)
        print('---------------------------')
        if i_batch > 100:
            break
    '''
    # test model
    model.eval()
    dataiter = iter(test_dataset_loader)
    images = dataiter.next()['input']
    # get sample outputs
    _, output = model(images)
    print(output.shape)
    # prep images for display
    images = images.numpy()
    # output is resized into a batch of iages
    output = output.view(batch_size, 1, 128, 128)
    # use detach when it's an output that requires_grad
    output = output.detach().numpy()

    # plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=2, ncols=batch_size, sharex=True, sharey=True, figsize=(25, 4))

    # input images on top row, reconstructions on bottom
    for images, row in zip([images, output], axes):
        for img, ax in zip(images, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.savefig('eval_result.png')
    '''



