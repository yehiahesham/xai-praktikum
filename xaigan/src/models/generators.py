from abc import ABC
from torch import nn, Tensor
import numpy as np



# BUG:check if it is ok have these edits, most likely not !
class GeneratorNetMSCOCO(nn.Module, ABC):
    def __init__(self,n_features=100):
        super(GeneratorNetMSCOCO, self).__init__()
        self.n_features = n_features
        #self.n_out = (3, 32, 32)
        self.n_out = (3, 256, 256)

        nc, nz, ngf = 3, n_features, 64

        self.input_layer = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 32, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),#nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
        )

        self.hidden1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),#nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
        )

        self.hidden2 = nn.Sequential(
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),#nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
        )

        self.out = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        #print("\tdense_emb/Gener Input shape is",x.shape)
        x = x[:,:, np.newaxis, np.newaxis]
        #print("\t1 maniplulation to that input, it becomes is",x.shape)
        x = self.input_layer(x)
        #print("output of Gen input_layer: ", x.size())
        x = self.hidden1(x)
        #print("Gen hidden1 layer: ", x.size())
        x = self.hidden2(x)
        #print("Gen hidden2 layer: ", x.size())
        x = self.out(x)
        #print("Gen out layer: ", x.size())
        return x



class GeneratorNet_TEXT2IMG_MSCOCO(nn.Module, ABC):
        def __init__(self,n_features=100):
            super(GeneratorNet_TEXT2IMG_MSCOCO, self).__init__()
            self.n_features = n_features
            #self.n_out = (3, 32, 32)
            size=256
            self.n_out = (3, size, size)
            nc, nz, ngf = 3, n_features, 64
            
            # size=256
            # s2, s4, s8, s16 = int(size/2), int(size/4), int(size/8), int(size/16)
            # self.n_features = n_features
            #d16, d8, d4, d2 = int(n_features*16), int(n_features*8), int(n_features* 4), int(n_features*2)
            d16,d8,d4,d2  = int(n_features/16), int(n_features/8), int(n_features/4), int(n_features/2)
            
            
            self.input_layer = nn.Sequential(
                nn.ConvTranspose2d(n_features, d2, 16, 1, 0, bias=False),
                nn.BatchNorm2d(d2),
                nn.ReLU(True),

            )

            self.hidden1 = nn.Sequential(
                nn.ConvTranspose2d(d2, d4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d4),
                nn.ReLU(True),
            )

            self.hidden2 = nn.Sequential(
                nn.ConvTranspose2d(d4,d8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d8),
                nn.ReLU(True),
            )

            self.hidden3 = nn.Sequential(
                nn.ConvTranspose2d(d8,d16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d16),
                nn.ReLU(True),
            )

            self.out = nn.Sequential(
                nn.ConvTranspose2d(d16, nc, 4, 2, 1, bias=False),
                nn.Tanh()
            )

        def forward(self, x):
            #print("\tdense_emb/Gener Input shape is",x.shape)
            x = x[:,:, np.newaxis, np.newaxis]
            #print("\t1 maniplulation to that input, it becomes is",x.shape)
            x = self.input_layer(x)
            x = self.hidden1(x)
            x = self.hidden2(x)
            x = self.hidden3(x)
            x = self.out(x)
            #print("Gen out layer: ", x.size())
            return x
