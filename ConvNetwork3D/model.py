import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super(Block, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=[3, 3, 3],
            stride=[1, 1, 1],
            padding=[1, 1, 1]
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=[3, 3, 3],
            stride=[2, 2, 2] if downsample else [1, 1, 1],
            padding=[1, 1, 1]
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.residual_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[1, 1, 1]),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(True)
        )

        if downsample:
            residual_conv_stride = [2, 2, 2]
        else:
            residual_conv_stride = [1, 1, 1]
            
        self.residual_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=[3, 3, 3], stride=residual_conv_stride, padding=[1, 1, 1]),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        residual = self.residual_conv(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class ConvNetwork(nn.Module):
    def __init__(self):
        super(ConvNetwork, self).__init__()

        self.layer1 = Block(1, 32, downsample=True)
        self.layer2 = Block(32, 64, downsample=True)
        self.layer3 = Block(64, 128, downsample=True)
        self.layer4 = Block(128, 256, downsample=False)

        self.last_layer = nn.Sequential(
            nn.Conv3d(256, 1, kernel_size=[1, 1, 1], stride=[1, 1, 1]),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # x = self.first_layer(x)
        
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)

        x = self.last_layer(x)
        # print(x.shape)

        return x
    
if __name__ == "__main__":
    DEVICE = "cuda"

    model = ConvNetwork().to(DEVICE)

    x = torch.rand([1, 1, 48, 64, 64]).to(DEVICE)
    y = model(x)
