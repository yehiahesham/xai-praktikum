# Implementation of the DCGAN paper : https://arxiv.org/abs/1511.06434

# Remove all pooling layers with strided convs
# Use batchnorm in both gen and disc
# remove all fc layers
# Use ReLU in gen except last, tanh
# use LeakyReLU in disc in all layers

import torch
import torch.nn as nn


# TODO: Modify the inputs for embeddings.

# DCGAN Discriminator Implementation
class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super().__init__()
        self.disc = nn.Sequential(
            # Input : N * channels_img * 64 * 64
            nn.Conv2d(
              channels_img, features_d, kernel_size=(4,), stride=(2,), padding=1 # TODO: Tuple?
            ), # 32*32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, kernel_size=4, stride=2, padding=1), # 16x16
            self._block(features_d * 2, features_d * 4, kernel_size=4, stride=2, padding=1), # 8x8
            self._block(features_d * 4, features_d * 8, kernel_size=4, stride=2, padding=1), # 4x4
            nn.Conv2d(features_d * 8, 1, kernel_size=(4,), stride=(2,), padding=0), # 1x1 single value fake or real
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      padding,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)

# DCGAN Generator Implementation
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input N x z_dim x 1 x 1. We'll pass features_g and d = 64 to make it 1024 like the paper.
            self._block(z_dim, features_g * 16, 4, 1, 0), # N x f_gx16 x 4 x 4
            self._block(features_g * 16, features_g * 8, 4, 2, 1), # 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1), # 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1), # 32x32
            nn.ConvTranspose2d(features_g * 2, channels_img, kernel_size=4, stride=2, padding=1), # 64x64
            nn.Tanh() # Images normalized btw [-1, 1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.gen(x)

# Weights inited with mean 0, std 0.02
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    N, in_channels, H, W = 8, 1, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator Assert Failed"
    gen = Generator(z_dim, in_channels, 8)
    initialize_weights(gen)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator Assert Failed"

#test()