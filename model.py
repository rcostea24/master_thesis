import torch

from septr.septr import SeparableTr
from conv_network_3d.conv_net import ConvNetwork

class Model(nn.Module):
    def __init__(self, septr_channels, septr_input_size, num_classes):
        super(Model, self).__init__()

        self.conv_network = ConvNetwork()
        self.transformer = SeparableTr(
            channels=septr_channels,
            input_size=septr_input_size,
            num_classes=num_classes
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