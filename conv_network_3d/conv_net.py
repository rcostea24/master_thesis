import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super(Block, self).__init__()

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=[3, 3, 3],
            stride=[2, 2, 2] if downsample else [1, 1, 1],
            padding=[1, 1, 1]
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ConvNetwork(nn.Module):
    def __init__(self):
        super(ConvNetwork, self).__init__()

        self.layer1 = Block(1, 4, downsample=True)
        self.layer2 = Block(4, 8, downsample=True)
        self.layer3 = Block(8, 16, downsample=True)
        self.layer4 = Block(16, 1, downsample=False)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
    
if __name__ == "__main__":
    DEVICE = "cuda"

    model = ConvNetwork().to(DEVICE)

    x = torch.rand([1, 1, 48, 64, 64]).to(DEVICE)
    y = model(x)
    print(torch.cuda.memory_allocated() * 10**(-6))
