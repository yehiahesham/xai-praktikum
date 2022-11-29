from abc import ABC
from torch import nn, Tensor
import numpy as np


class DiscriminatorNetMSCOCO(nn.Module, ABC):
    def __init__(self):
        super(DiscriminatorNetMSCOCO, self).__init__()
        self.n_features = (3, 256, 256)
        nc, ndf = 3, 64

        self.input_layer = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.hidden1 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.hidden2 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.out = nn.Sequential(
            nn.Conv2d(ndf * 4, 1, 32, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        x = self.input_layer(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x