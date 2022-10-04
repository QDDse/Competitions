import torch
import torch.nn as nn
import timm
from torchvision import models as M

## Effnet_baseline
class Effnet(nn.Module):
    def __init__(self, in_c=3, num_class=2, model_name = None):
        super(Effnet, self).__init__()
        self.in_c = in_c
        self.num_class = num_class

        self.model = timm.create_model(model_name=model_name, pretrained=True)
        # classifier = nn.Linear(2304, num_class)
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=2304, out_features=2, bias=True)
        )
        # self.model.classifier = classifier
    def forward(self, x):
        return self.model(x)
    
class Swin_trm(nn.Module):
    def __init__(self, in_c=3, num_class=2, model_name = None):
        super(Swin_trm, self).__init__()
        self.in_c = in_c
        self.num_class = num_class

        self.model = timm.create_model(model_name=model_name, pretrained=True)
        # classifier = nn.Linear(2304, num_class)
        in_features = self.model.head.in_features
        self.model.head = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=in_features, out_features=num_class, bias=True)
        )
        # self.model.classifier = classifier
    def forward(self, x):
        return self.model(x)