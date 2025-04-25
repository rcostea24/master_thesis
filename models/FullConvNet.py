import numpy as np
import torch
from torch import nn

from models.ResNet3D import ResNet3D

class FullConvNet(nn.Module):
    def __init__(self, resnet_params, num_classes):
        super(FullConvNet, self).__init__()

        self.resnet3d = ResNet3D(**resnet_params)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.mean(dim=1, keepdim=True)
        features = self.resnet3d(x)
        out = self.classifier(features)
        return out
    
if __name__ == "__main__":
    x = torch.rand([32, 32, 32, 24, 70]).to("cuda")

    x = x.view(-1, 70)

    pool = nn.AvgPool1d(kernel_size=70)
    x = pool(x.unsqueeze(1))

    x = x.view(32, 1, 32, 32, 24)

    resnet_params = {
        "layers": [2, 2, 2, 2]
    }
    
    model = FullConvNet(resnet_params, 3).to("cuda")
    y = model(x)
    print(y.shape)