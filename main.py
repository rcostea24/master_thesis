import torch
import numpy as np
import torch.nn as nn
import nibabel as nib

from septr.septr import SeparableTr
from ConvNetwork3D.conv_net import ConvNetwork

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv_network = ConvNetwork()
        self.transformer = SeparableTr(
            channels=6,
            input_size=(64, 10),
            num_classes=2
        )

    def forward(self, x):
        downsampled_volumes = []
        for id in range(10):
            volume_out = torch.squeeze(
                self.conv_network(
                    torch.unsqueeze(x[:, :, :, :, id], dim=1)
                ), 
                dim=1)

            b, d1, d2, d3 = volume_out.shape
            volume_out = torch.reshape(volume_out, [b, d1, d2*d3])

            downsampled_volumes.append(volume_out)

        out = self.transformer(torch.stack(downsampled_volumes, dim=-1))
        return out

if __name__ == "__main__":
    DEVICE = "cuda"

    model = Model().to(DEVICE)

    x = torch.rand([1, 48, 64, 64, 70]).to(DEVICE)
    y = model(x)
    print(y.shape)
    print(torch.cuda.memory_allocated() * 10**(-6))