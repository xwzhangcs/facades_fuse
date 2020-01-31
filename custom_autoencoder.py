from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models.resnet import BasicBlock, conv1x1, conv3x3
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class AutoEncoder(nn.Module):
    def __init__(self, in_channels, dec_channels, latent_size):
        super(AutoEncoder, self).__init__()

        self.in_channels = in_channels
        self.dec_channels = dec_channels
        self.latent_size = latent_size

        ###############
        # ENCODER
        ##############
        self.e_conv_1 = nn.Conv2d(in_channels, dec_channels,
                                  kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.e_bn_1 = nn.BatchNorm2d(dec_channels)

        self.e_conv_2 = nn.Conv2d(dec_channels, dec_channels * 2,
                                  kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.e_bn_2 = nn.BatchNorm2d(dec_channels * 2)

        self.e_conv_3 = nn.Conv2d(dec_channels * 2, dec_channels * 4,
                                  kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.e_bn_3 = nn.BatchNorm2d(dec_channels * 4)

        self.e_conv_4 = nn.Conv2d(dec_channels * 4, dec_channels * 8,
                                  kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.e_bn_4 = nn.BatchNorm2d(dec_channels * 8)

        self.e_conv_5 = nn.Conv2d(dec_channels * 8, dec_channels * 16,
                                  kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.e_bn_5 = nn.BatchNorm2d(dec_channels * 16)

        self.e_fc_1 = nn.Linear(dec_channels * 16 * 4 * 4, latent_size)

        ###############
        # DECODER
        ##############

        self.d_fc_1 = nn.Linear(latent_size, dec_channels * 16 * 4 * 4)

        self.d_conv_1 = nn.Conv2d(dec_channels * 16, dec_channels * 8,
                                  kernel_size=(4, 4), stride=(1, 1), padding=0)
        self.d_bn_1 = nn.BatchNorm2d(dec_channels * 8)

        self.d_conv_2 = nn.Conv2d(dec_channels * 8, dec_channels * 4,
                                  kernel_size=(4, 4), stride=(1, 1), padding=0)
        self.d_bn_2 = nn.BatchNorm2d(dec_channels * 4)

        self.d_conv_3 = nn.Conv2d(dec_channels * 4, dec_channels * 2,
                                  kernel_size=(4, 4), stride=(1, 1), padding=0)
        self.d_bn_3 = nn.BatchNorm2d(dec_channels * 2)

        self.d_conv_4 = nn.Conv2d(dec_channels * 2, dec_channels,
                                  kernel_size=(4, 4), stride=(1, 1), padding=0)
        self.d_bn_4 = nn.BatchNorm2d(dec_channels)

        self.d_conv_5 = nn.Conv2d(dec_channels, in_channels,
                                  kernel_size=(4, 4), stride=(1, 1), padding=0)

        # Reinitialize weights using He initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.detach())
                m.bias.detach().zero_()
            elif isinstance(m, torch.nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.detach())
                m.bias.detach().zero_()
            elif isinstance(m, torch.nn.Linear):
                nn.init.kaiming_normal_(m.weight.detach())
                m.bias.detach().zero_()

    def encode(self, x):

        # h1
        x = self.e_conv_1(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.e_bn_1(x)

        # h2
        x = self.e_conv_2(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.e_bn_2(x)

        # h3
        x = self.e_conv_3(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.e_bn_3(x)

        # h4
        x = self.e_conv_4(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.e_bn_4(x)

        # h5
        x = self.e_conv_5(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.e_bn_5(x)

        # fc
        x = x.view(-1, self.dec_channels * 16 * 4 * 4)
        x = self.e_fc_1(x)
        return x

    def decode(self, x):

        # h1
        # x = x.view(-1, self.latent_size, 1, 1)
        x = self.d_fc_1(x)

        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = x.view(-1, self.dec_channels * 16, 4, 4)

        # h2
        x = F.interpolate(x, scale_factor=2)
        x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
        x = self.d_conv_1(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.d_bn_1(x)

        # h3
        x = F.interpolate(x, scale_factor=2)
        x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
        x = self.d_conv_2(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.d_bn_2(x)

        # h4
        x = F.interpolate(x, scale_factor=2)
        x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
        x = self.d_conv_3(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.d_bn_3(x)

        # h5
        x = F.interpolate(x, scale_factor=2)
        x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
        x = self.d_conv_4(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.d_bn_4(x)

        # out
        x = F.interpolate(x, scale_factor=2)
        x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
        x = self.d_conv_5(x)
        x = torch.sigmoid(x)

        return x

    def forward(self, x1, x2):
        print('forward x1: ', x1.shape)
        print('forward x2: ', x2.shape)
        z1 = self.encode(x1)
        z2 = self.encode(x2)
        z = z1.add_(z2)
        z = z / 2
        print(z.shape)
        decoded = self.decode(z)
        return z, decoded







