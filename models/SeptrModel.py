import numpy as np
import torch
from torch import nn
from models.SeptrConvNet import ConvNetwork
from septr.septr import SeparableTr

class SeptrModel(nn.Module):
    def __init__(self, septr_params, num_classes):
        super(SeptrModel, self).__init__()
        septr_params["input_size"] = tuple(septr_params["input_size"])
        self.conv_network = ConvNetwork()
        self.transformer = SeparableTr(num_classes=num_classes, **septr_params)

    def forward(self, x):
        time_stamps = np.arange(x.shape[1])
        downsampled_volumes = []
        for time_stamp in time_stamps:
            volume_in = torch.unsqueeze(x[:, time_stamp, :, :, :], dim=1)
            volume_out = torch.squeeze(self.conv_network(volume_in), dim=1)

            b, d1, d2, d3 = volume_out.shape
            volume_out = torch.reshape(volume_out, [b, d1, d2*d3])

            downsampled_volumes.append(volume_out)

        downsampled_volumes = torch.stack(downsampled_volumes, dim=1)
        out, _ = self.transformer(downsampled_volumes)
        return out
    
if __name__ == "__main__":
    x = torch.rand([32, 32, 32, 24, 70]).to("cuda")
    model = SeptrModel(70, (8, 48), 6).to("cuda")
    y = model(x)
    print(y.shape)