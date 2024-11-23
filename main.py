import torch
import numpy as np
import torch.nn as nn
import nibabel as nib

from septr.septr import SeparableTr
from ConvNetwork3D.model import ConvNetwork

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv_network = ConvNetwork()
        self.transformer = SeparableTr(
            channels=6,
            input_size=(64, 140),
            num_classes=2
        )

    def forward(self, x):
        downsampled_input = torch.empty([x.shape[0], 6, 64, x.shape[-1]]).to(DEVICE)
        for id in range(x.shape[-1]):
            volume_in = x[:, :, :, :, id]
            volume_in = torch.unsqueeze(volume_in, dim=1)
            volume_out = self.conv_network(volume_in)
            volume_out = torch.squeeze(volume_out, dim=1)

            b, d1, d2, d3 = volume_out.shape
            volume_out = torch.reshape(volume_out, [b, d1, d2*d3])

            downsampled_input[:, :, :, id] = volume_out

        out = self.transformer(downsampled_input)
        return out

if __name__ == "__main__":
    DEVICE = "cpu"

    model = Model().to(DEVICE)
    print(model)

    x = torch.rand([1, 48, 64, 64, 140]).to(DEVICE)
    # y = model(x)
    # print(y.shape)