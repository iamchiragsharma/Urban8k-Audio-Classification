from urban_base_model import UrbanSoundBase

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

class NNModel_3Layers(UrbanSoundBase):
    """
    Layer Size : 128, 256, 10
    Drop Outs : True, True, True
    Drop Values : 0.2, 0.3, 0.2
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2))
        
        self.block2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3))
        
        self.block3 = nn.Sequential(
            nn.Linear(256, output_size),
            nn.ReLU(),
            nn.Dropout(0.3))

        
    def forward(self, xb):
        out = self.block1(xb)
        out = self.block2(out)
        return self.block3(out)