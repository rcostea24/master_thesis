import torch
from torch import nn
from conv_network_3d.conv_net import ConvNetwork
from septr.septr import SeparableTr

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
        #TODO: Permute input such that timestamps are on dim=1
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
    x = torch.rand([32, 64, 64, 48, 140]).to("cuda")
    model = Model(140, (8, 48), 6).to("cuda")
    y = model(x)
    print(y.shape)