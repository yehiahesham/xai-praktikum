from abc import ABC
from torch import nn, Tensor
import numpy as np

    
class EmbeddingEncoderNetMSCOCO(nn.Module, ABC):
    def __init__(self,noise_emb_sz,text_emb_sz,generator_input_emb_sz=100):
        super(EmbeddingEncoderNetMSCOCO, self).__init__()
        self.input_features  = noise_emb_sz+text_emb_sz
        self.output_features = generator_input_emb_sz
        ngf = 64

        self.input_layer = nn.Sequential(
            nn.Linear(self.input_features, ngf * 8, bias=False),
            nn.BatchNorm1d(ngf * 8),
            nn.ReLU(True),
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(ngf * 8, ngf * 4,bias=False),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True),
        )

        self.hidden2 = nn.Sequential(
            nn.Linear( ngf * 4, ngf * 2, bias=False),
            nn.BatchNorm1d(ngf * 2),
            nn.ReLU(True),
        )

        self.out = nn.Sequential(
            nn.Linear(ngf * 2, self.output_features,  bias=False),
            nn.ReLU(True)
        )

    def forward(self, x):
        
        x = self.input_layer(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
