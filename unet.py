import torch
import numpy as np
import torch.nn as nn

class CBR_block(nn.Module):
  def __init__(self, in_channels, out_channels, use_norm=True):
    super(CBR_block, self).__init__()

    if (use_norm == True):
        self.use_norm = True
        self.conv1 = nn.Conv2d(
                     in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=3,
                     padding=1,
                     bias=False
                     )
        self.batchnorm1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
                     in_channels=out_channels,
                     out_channels=out_channels,
                     kernel_size=3,
                     padding=1,
                     bias=False
                     )
        self.batchnorm2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu2 = nn.ReLU(inplace=True)
    else:
        self.use_norm = False
        self.conv1 = nn.Conv2d(
                     in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=3,
                     padding=1,
                     bias=False
                     )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
                     in_channels=out_channels,
                     out_channels=out_channels,
                     kernel_size=3,
                     padding=1,
                     bias=False
                     )
        self.relu2 = nn.ReLU(inplace=True)

  def forward(self, x):
    next = self.conv1(x)
    if self.use_norm:
        next = self.batchnorm1(next)
    next = self.relu1(next)
    next = self.conv2(next)
    if self.use_norm:
        next = self.batchnorm2(next)
    return self.relu2(next)

class Unet(nn.Module):
  def __init__(self, in_channels, out_channels, use_residual=True, use_norm=True):
    super(Unet, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.use_residual = use_residual

    self.inc1 = CBR_block(in_channels, 32, use_norm=use_norm)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.inc2 = CBR_block(32, 64, use_norm=use_norm)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.inc3 = CBR_block(64, 128, use_norm=use_norm)
    self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.middle = CBR_block(128, 256, use_norm=use_norm)
    self.unpool3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
    self.dec3 = CBR_block(256, 128, use_norm=use_norm)
    self.unpool2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
    self.dec2 = CBR_block(128, 64, use_norm=use_norm)
    self.unpool1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
    self.dec1 = CBR_block(64, 32, use_norm=use_norm)
    self.one_by_one_conv = nn.Conv2d(in_channels=32,
                           out_channels=out_channels,
                           kernel_size=1,
                           padding=0,
                           bias=False)
    
  def forward(self, x):
    i1 = self.inc1(x)
    i2 = self.inc2(self.pool1(i1))
    i3 = self.inc3(self.pool2(i2))
    mid = self.middle(self.pool3(i3))
    d3 = self.dec3(torch.cat((self.unpool3(mid), i3), dim=1))
    d2 = self.dec2(torch.cat((self.unpool2(d3), i2), dim=1))
    d1 = self.dec1(torch.cat((self.unpool1(d2), i1), dim=1))
    block = self.one_by_one_conv(d1)
    if self.use_residual:
      return block + x
    else:
      return block