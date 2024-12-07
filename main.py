import torch
import numpy as np
import torch.nn as nn
import nibabel as nib

from septr.septr import SeparableTr
from conv_network_3d.conv_net import ConvNetwork

DEVICE = "cuda"

INPUT_SIZE = (6, 64)
CHANNELS = 140
NUM_CLASSES = 6
BATCH_SIZE = 32

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv_network = ConvNetwork()
        self.transformer = SeparableTr(
            channels=CHANNELS,
            input_size=INPUT_SIZE,
            num_classes=NUM_CLASSES
        )

    def forward(self, x):
        downsampled_volumes = []
        for id in range(x.shape[-1]):
            volume_in = torch.unsqueeze(x[:, :, :, :, id], dim=1)
            volume_out = torch.squeeze(self.conv_network(volume_in), dim=1)

            b, d1, d2, d3 = volume_out.shape
            volume_out = torch.reshape(volume_out, [b, d1, d2*d3])

            downsampled_volumes.append(volume_out)

        downsampled_volumes = torch.stack(downsampled_volumes, dim=1)
        out = self.transformer(downsampled_volumes)
        return out

if __name__ == "__main__":
    model = Model().to(DEVICE)

    x = torch.rand([BATCH_SIZE, 48, 64, 64, 140]).to(DEVICE)
    y = model(x)
    print(torch.cuda.memory_allocated() * 10**(-6))