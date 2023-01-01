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
        #print("x before input layer", x.size())
        x = self.input_layer(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x




class DiscriminatorNet_TEXT2IMG_MSCOCO(nn.Module):
    def __init__(self):
        super(DiscriminatorNet_TEXT2IMG_MSCOCO, self).__init__()
        self.n_features = (3, 256, 256)
        nc, ndf = 3, 256/8
        
        # Took as original as 2400 don't know why
        caption_length = 2400
        t_dim = 256

        # Kernel 5x5, dilation=2?, stride=2, nc=3, ndf=32
        self.hidden1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )
        self.hidden2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

        self.hidden3 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU() 
        )

        self.hidden4 = nn.Sequential(
            nn.Conv2d(128, 256, 5, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        self.out = nn.Sequential(
	    nn.Conv2d(256, 1, 15, 1, 0, bias=False),
	    nn.Sigmoid()
	)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.out(x)
        return x