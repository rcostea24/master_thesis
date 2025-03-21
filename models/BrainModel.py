import numpy as np
import torch
from torch import nn

from ResNet3D import ResNet3D
from Transformer import Transformer

class Model(nn.Module):
    def __init__(self, resnet_params, transformer_params):
        super(Model, self).__init__()

        self.resnet3d = ResNet3D(**resnet_params)
        self.transformer = Transformer(**transformer_params)
        self.classifier = nn.Linear(512, 3)

    def forward(self, x):
        x = torch.permute(x, (0, 4, 1, 2, 3))
        time_stamps = np.arange(x.shape[1])
        features = []
        for time_stamp in time_stamps:
            volume_in = torch.unsqueeze(x[:, time_stamp, :, :, :], dim=1)
            out_features = torch.squeeze(self.resnet3d(volume_in), dim=1)
            features.append(out_features)

        features = torch.stack(features, dim=1)
        transformer_out = self.transformer(features)
        transformer_out = transformer_out.mean(dim=1)
        
        out = self.classifier(transformer_out)
        
        return out
    
if __name__ == "__main__":
    x = torch.rand([32, 32, 32, 24, 70]).to("cuda")
    model = Model([2, 2, 2, 2]).to("cuda")
    y = model(x)
    print(y.shape)