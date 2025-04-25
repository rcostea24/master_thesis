import numpy as np
import torch
from torch import nn
from models.SeptrConvNet import ConvNetwork
from septr.septr import SeparableTr

class SeptrInParallel(nn.Module):
    def __init__(self, params, num_classes):
        super(SeptrInParallel, self).__init__()
        self.conv_network = ConvNetwork()
        self.transformers = nn.ModuleList()
        
        dims = params["input_dims"]
        input_sizes = [
            (dims[0]*dims[1], dims[2]),
            (dims[0]*dims[2], dims[1]),
            (dims[0], dims[1]*dims[2])
        ]
        
        for input_size in input_sizes:
            self.transformers.append(
                SeparableTr(
                    input_size=input_size,
                    num_classes=num_classes,
                    **params["septr_params"]
                )
            )
            
        self.linear = nn.Linear(params["septr_params"]["dim"], num_classes)

    def forward(self, x):
        time_stamps = np.arange(x.shape[1])
        downsampled_volumes = [[] for _ in range(3)]
        for time_stamp in time_stamps:
            volume_in = x[:, time_stamp, :, :, :].unsqueeze(dim=1)
            volume_out = self.conv_network(volume_in).squeeze(dim=1)

            b, h, w, d = volume_out.shape
        
            downsampled_volumes[0].append(volume_out.reshape([b, h*w, d]))
            downsampled_volumes[1].append(volume_out.reshape([b, h*d, w]))
            downsampled_volumes[2].append(volume_out.reshape([b, h, w*d]))

        cls_tokens = []
        for input, transformer in zip(downsampled_volumes, self.transformers):
            input_tensor = torch.stack(input, dim=1)
            _, cls_token = transformer(input_tensor)
            cls_tokens.append(cls_token)
            
        cls_tokens = torch.stack(cls_tokens, dim=1)
        cls_tokens = cls_tokens.mean(dim=1)
        output = self.linear(cls_tokens)
            
        return output
    
if __name__ == "__main__":
    x = torch.rand([32, 32, 32, 24, 70]).to("cuda")
    params = {
        "input_dims": [8, 8, 6],
        "septr_params": {
            "channels": 70,
            "num_classes": 3, 
            "depth": 3, 
            "heads": 5, 
            "mlp_dim": 256, 
            "dim_head": 256, 
            "dim": 256,
            "dropout_tr": 0.2
        }
    }
    model = SeptrInParallel(params).to("cuda")
    y = model(x)
    print(y.shape)