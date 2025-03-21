import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class ResNet3D(nn.Module):
    def __init__(self, layers):
        super(ResNet3D, self).__init__()
        
        self.in_channels = 64
        
        # Initial Conv Layer
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Layers of residual blocks
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        
        # Fully connected layer
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
    
    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels),
            )
        
        layers = []
        layers.append(BasicBlock3D(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x

if __name__ == "__main__":
    # Example: ResNet-18 for 3D data
    sample = torch.rand([32, 70, 32, 32, 24])
    model = ResNet3D([2, 2, 2, 2])
    
    x = sample[:, 0, :, :, :].unsqueeze(dim=1)
    y = model(x)
    print(y.shape)
