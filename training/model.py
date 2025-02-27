import torch
import torch.nn as nn

import importlib
class Model(nn.Module):
    # main model class
    def __init__(self, vision_params, language_params, classifier_params):
        super(Model, self).__init__()

        # import vision model specified in cfg
        vision_module = importlib.import_module(f"vision_model.{vision_params['model_type']}")
        vision_model_class = getattr(vision_module, "VisionModel")

        # init vision model
        self.vision_model = vision_model_class(vision_params)

        # load pretrained weights if they exists
        if "weights" in vision_params and vision_params["weights"] is not None:
            state_dict = torch.load(vision_params["weights"], map_location="cpu")
            backbone_state_dict = {}
            for key, value in state_dict.items():
                if key in self.vision_model.state_dict():
                    backbone_state_dict[key] = value
            self.vision_model.load_state_dict(backbone_state_dict)
            print("vision weights loaded")

        # import language model specified in cfg
        language_module = importlib.import_module(f"language_model.{language_params['model_type']}")
        language_model_class = getattr(language_module, "LanguageModel")
        
        # init language model
        self.language_model = language_model_class(language_params)

        # load pretrained weights if they exists
        if "weights" in language_params and language_params["weights"] is not None:
            state_dict = torch.load(language_params["weights"])
            backbone_state_dict = {}
            for key, value in state_dict.items():
                if key in self.language_model.state_dict():
                    backbone_state_dict[key] = value
            self.language_model.load_state_dict(backbone_state_dict)
            print("language weights loaded")

        # get classifier's layers
        fc_size = classifier_params["fc_size"]
        
        # get classifier's activation function
        cls_act = getattr(nn, classifier_params["act"])

        # add first layer 
        cls_layers = [
            nn.BatchNorm1d(fc_size[0]),
            nn.Linear(
                fc_size[0], 
                fc_size[1]
            ),
            cls_act()
        ]

        # add the rest of the layers
        for id in range(1, len(fc_size)-1):
            cls_layers.append(nn.Linear(fc_size[id], fc_size[id+1]))
            
            if id < len(fc_size) - 2:
                cls_layers.append(cls_act())

        # set them into sequential module
        self.classifier = nn.Sequential(*cls_layers)

    def forward(self, img_x, txt_x):
        # forward pass

        # get image representation
        vision_embed = self.vision_model(img_x)

        if isinstance(vision_embed, list):
            # if output of vision model is a list
            # then it is an autoencoder and only the bottleneck features should be used
            vision_embed = vision_embed[1]

            # flatten the tensor
            dim1 = vision_embed.shape[0]
            dim2 = vision_embed.shape[1] * vision_embed.shape[2] * vision_embed.shape[3]
            vision_embed = vision_embed.view(dim1, dim2)

        # get caption representation
        language_embed = self.language_model(txt_x)

        # concatenate image-text features
        classifier_input = torch.cat([vision_embed, language_embed], dim=1)

        # get the output of fc classifier
        out = self.classifier(classifier_input)

        return out