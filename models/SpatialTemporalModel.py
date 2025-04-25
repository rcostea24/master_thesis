import numpy as np
import torch
from torch import nn

from models.ResNet3D import ResNet3D
from models.GRU import GRU

class SpatialTemporalModel(nn.Module):
    def __init__(self, params, num_classes):
        super(SpatialTemporalModel, self).__init__()
        self.spatial_model = ResNet3D(**params["spatial_model"])
        self.temporal_model = GRU(**params["temporal_model"])
        
        clf_input_size = params["temporal_model"]["hidden_dim"]
        if params["temporal_model"]["bidirectional"]:
            clf_input_size *= 2
            
        self.classifier = nn.Sequential(
            nn.Linear(clf_input_size, num_classes)
        )

    def forward(self, x):
        time_stamps = np.arange(x.shape[1])
        spatial_features = []
        for time_stamp in time_stamps:
            volume_in = torch.unsqueeze(x[:, time_stamp, :, :, :], dim=1)
            spatial_features.append(
                torch.squeeze(self.spatial_model(volume_in), dim=1)
            )

        spatial_features = torch.stack(spatial_features, dim=1)
        temporal_features = self.temporal_model(spatial_features)
        output = self.classifier(temporal_features)
        return output