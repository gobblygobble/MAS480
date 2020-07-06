import torch
import numpy as np
import torch.nn as nn

class CBR_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CBR_block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu2 = nn.ReLU(inplace=True)
   
    def forward(self, x):
        next = self.conv1(x)
        next = self.batchnorm1(next)
        next = self.relu1(next)
        next = self.conv2(next)
        next = self.batchnorm2(next)
        return self.relu2(next)

class Attention_block(nn.Module):
    def __init__(self, F_g, F_x):
        super(Attention_block, self).__init__()
        # in our example, F_g and F_x should always be same
        # and F_int is always half of F_g (or F_x)
        assert F_g == F_x
        #F_g = int(F_g)
        #F_x = int(F_x)
        F_int = F_g // 2
        # for block W_g
        self.W_g_conv = nn.Conv2d(in_channels=F_g, out_channels=F_int,
                                  kernel_size=1, padding=0, bias=False)
        self.W_g_batchnorm = nn.BatchNorm2d(num_features=F_int)
        # for block W_x
        self.W_x_conv = nn.Conv2d(in_channels=F_x, out_channels=F_int,
                                  kernel_size=1, padding=0, bias=False)
        self.W_x_batchnorm = nn.BatchNorm2d(num_features=F_int)
        # for block ReLU
        self.relu = nn.ReLU(inplace=True)
        # for block psi
        # TODO: out_channels not mentioned in paper diagram... 1?
        self.psi_conv = nn.Conv2d(in_channels=F_int, out_channels=1,
                                  kernel_size=1, padding=0, bias=False)
        # TODO: if out_channel changes, change this as well
        self.psi_batchnorm = nn.BatchNorm2d(num_features=1)
        # for block sigmoid
        self.sigmoid = nn.Sigmoid()
        # NO resampling
    def forward(self, g, x):
        # block W_g
        next_g = self.W_g_conv(g)
        next_g = self.W_g_batchnorm(next_g)
        # block W_x
        next_x = self.W_x_conv(x)
        next_x = self.W_x_batchnorm(next_x)
        # rest of the blocks - in order
        relu = self.relu(next_g + next_x)
        next_psi = self.psi_conv(relu)
        #next_psi = self.psi_batchnorm(next_psi)
        alpha = self.sigmoid(next_psi)
        return alpha * x


class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()
        # encoding paths
        self.inc1 = CBR_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.inc2 = CBR_block(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.inc3 = CBR_block(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # decoding paths
        self.middle = CBR_block(128, 256)
        self.unpool3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = CBR_block(256, 128)
        self.unpool2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = CBR_block(128, 64)
        self.unpool1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = CBR_block(64, 32)
        self.one_by_one_conv = nn.Conv2d(in_channels=32, out_channels=out_channels,
                                         kernel_size=1, padding=0, bias=False)
    
    def forward(self, x):
        i1 = self.inc1(x)
        i2 = self.inc2(self.pool1(i1))
        i3 = self.inc3(self.pool2(i2))
        mid = self.middle(self.pool3(i3))
        d3 = self.dec3(torch.cat((self.unpool3(mid), i3), dim=1))
        d2 = self.dec2(torch.cat((self.unpool2(d3), i2), dim=1))
        d1 = self.dec1(torch.cat((self.unpool1(d2), i1), dim=1))
        block = self.one_by_one_conv(d1)
        return block

class AttentionUnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionUnet, self).__init__()
        # encoding paths
        self.inc1 = CBR_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.inc2 = CBR_block(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.inc3 = CBR_block(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.middle = CBR_block(128, 256)
        # decoding paths
        self.unpool3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.attblock3 = Attention_block(F_g=128, F_x=128)
        self.dec3 = CBR_block(256, 128)
        self.unpool2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.attblock2 = Attention_block(F_g=64, F_x=64)
        self.dec2 = CBR_block(128, 64)
        self.unpool1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.attblock1 = Attention_block(F_g=32, F_x=32)
        self.dec1 = CBR_block(64, 32)
        self.one_by_one_conv = nn.Conv2d(in_channels=32, out_channels=out_channels,
                                         kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        i1 = self.inc1(x)
        i2 = self.inc2(self.pool1(i1))
        i3 = self.inc3(self.pool2(i2))
        mid = self.middle(self.pool3(i3))
        d3temp = self.unpool3(mid)
        # dim=1 because NxCxHxW and we want to concat C-wise
        d3 = self.dec3(torch.cat((self.attblock3(g=d3temp, x=i3), d3temp), dim=1))
        d2temp = self.unpool2(d3)
        d2 = self.dec2(torch.cat((self.attblock2(g=d2temp, x=i2), d2temp), dim=1))
        d1temp = self.unpool1(d2)
        d1 = self.dec1(torch.cat((self.attblock1(g=d1temp, x=i1), d1temp), dim=1))
        block = self.one_by_one_conv(d1)
        return block