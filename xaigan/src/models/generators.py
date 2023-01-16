from abc import ABC
from torch import nn, Tensor
import torch
import numpy as np

#============================================ start of ENCODERS =======================================================================================

class EmbeddingEncoderNetMSCOCO(nn.Module, ABC):
    def __init__(self,noise_emb_sz,text_emb_sz=868,generator_input_emb_sz=100):
        super(EmbeddingEncoderNetMSCOCO, self).__init__()
        self.input_features  = noise_emb_sz+text_emb_sz
        self.output_features = generator_input_emb_sz
        ngf = 64

        self.input_layer = nn.Sequential(
            nn.Linear(self.input_features, ngf * 8, bias=False),
            nn.BatchNorm1d(ngf * 8),
            nn.ReLU(True),
            nn.Dropout(0.2)
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(ngf * 8, ngf * 4,bias=False),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True),
            nn.Dropout(0.2)
        )

        self.hidden2 = nn.Sequential(
            nn.Linear( ngf * 4, ngf * 2, bias=False),
            nn.BatchNorm1d(ngf * 2),
            nn.ReLU(True),
            nn.Dropout(0.2)
        )

        self.out = nn.Sequential(
            nn.Linear(ngf * 2, self.output_features,  bias=False),
            nn.ReLU(True),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        
        x = self.input_layer(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

#============================================ start of Generators =====================================================================================

class GeneratorNetCIFAR10(nn.Module, ABC):
    def __init__(self,n_features=100):
        super(GeneratorNetCIFAR10, self).__init__()
        self.n_features = n_features
        self.n_out = (3, 32, 32)

        nc, nz, ngf = 3, n_features, 64

        self.input_layer = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
        )

        self.hidden1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
        )

        self.hidden2 = nn.Sequential(
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
        )

        self.out = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x



class GeneratorNet_MSCOCO(nn.Module, ABC):
    def __init__(self,n_features=100):
        super(GeneratorNet_MSCOCO, self).__init__()
        self.n_features = n_features
        self.n_out = (3, 256, 256)

        nc, nz, ngf = 3, n_features, 64

        self.input_layer = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 32, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
        )

        self.hidden1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
        )

        self.hidden2 = nn.Sequential(
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
        )

        self.out = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

class GeneratorNet_2_MSCOCO(nn.Module, ABC):
    def __init__(self,n_features=100):
        super(GeneratorNet_2_MSCOCO, self).__init__()
        self.n_features = n_features
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
        x = self.input_layer(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.out(x)
        return x


#============================================ start of Encoders + Generators =====================================================================================

#====== start of Encoders + Generators (MSCOCO sizes) ======
class Generator_Encoder_Net_MSCOCO(nn.Module, ABC):
    def __init__(self,noise_emb_sz=100,text_emb_sz=768,n_features=868):
        super(Generator_Encoder_Net_MSCOCO, self).__init__()
        
        self.noise_emb_sz   = noise_emb_sz
        self.text_emb_sz    = text_emb_sz
        self.generator_feat = n_features
                
        self.encoder =   EmbeddingEncoderNetMSCOCO(noise_emb_sz,text_emb_sz,n_features)
        self.generator = GeneratorNet_MSCOCO(n_features=n_features)
        
        
    def forward(self, x):
        x = self.encoder(x)
        x = x[:,:, np.newaxis, np.newaxis] #add new access to pass to generator
        x = self.generator(x)
        return x

class GeneratorNet_2_EncoderNet_MSCOCO(nn.Module, ABC):
    def __init__(self,noise_emb_sz=100,text_emb_sz=768,n_features=868):
        super(GeneratorNet_2_EncoderNet_MSCOCO, self).__init__()
        
        self.noise_emb_sz   = noise_emb_sz
        self.text_emb_sz    = text_emb_sz
        self.generator_feat = n_features
        
        
        self.encoder   = EmbeddingEncoderNetMSCOCO(noise_emb_sz,text_emb_sz,n_features)
        self.generator = GeneratorNet_2_MSCOCO(n_features=n_features)
        
        
    def forward(self, x):
        x = self.encoder(x)
        x = x[:,:, np.newaxis, np.newaxis] #add new access to pass to generator
        x = self.generator(x)
        return x

#====== start of Encoders + Generators (CIFAR-10 sizes) ======

class Generator_Encoder_Net_CIFAR10(nn.Module, ABC):
    def __init__(self,noise_emb_sz=100,text_emb_sz=768,n_features=868):
        super(Generator_Encoder_Net_CIFAR10, self).__init__()
        
        self.noise_emb_sz   = noise_emb_sz
        self.text_emb_sz    = text_emb_sz
        self.generator_feat = n_features
        
        
        self.encoder = EmbeddingEncoderNetMSCOCO(noise_emb_sz,text_emb_sz,n_features)
        self.generator = GeneratorNetCIFAR10(n_features=n_features)
        
        
    def forward(self, x):
        x = self.encoder(x)
        x = x[:,:, np.newaxis, np.newaxis] #add new access to pass to generator
        x = self.generator(x)
        return x

