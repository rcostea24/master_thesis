import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, in_features, num_heads_per_layer, num_layers, layer_params={}):
        super().__init__()
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_features,
            nhead=num_heads_per_layer,
            **layer_params
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        return self.transformer_encoder(x)
    
if __name__ == "__main__":
    x = torch.rand([32, 512]).to("cuda")
    transformer = Transformer(512, 4, 3).to("cuda")
    y = transformer(x)
    print(y.shape)