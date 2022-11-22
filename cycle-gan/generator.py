# Imports
import torch
import torch.nn as nn


# Block of Convolutional Layers
class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsampling=True, use_activations=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if downsampling
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_activations else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)

# Block of Residuals
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.residual = nn.Sequential(
            ConvolutionalBlock(channels, channels, kernel_size=3, padding=1,stride=1),
            ConvolutionalBlock(channels, channels, use_activations=False, kernel_size=3, padding=1)
        )
    def forward(self, x):
        return x + self.residual(x)

# Generator Implementation
# num_res = 9 if 256x256
# num_res = 6 if 128x128 or smaller

class Generator(nn.Module):
    def __init__(self, image_channels, num_of_features=64, num_residuals=9):
        super().__init__()
        # Convolutional Block without InstanceNorm.
        self.init = nn.Sequential(
            nn.Conv2d(image_channels, num_of_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.ReLU(inplace=True)
        )

        self.downblocks = nn.ModuleList(
            [
                ConvolutionalBlock(num_of_features, num_of_features*2, downsampling=True, kernel_size=3, stride=2, padding=1),
                ConvolutionalBlock(num_of_features*2, num_of_features*4, downsampling=True, kernel_size=3, stride=2, padding=1)
            ]
        )

        self.residuals = nn.Sequential(
            *[ResidualBlock(num_of_features*4) for _ in range(num_residuals)] # 9 Residual Blocks
        )

        self.upsample_blocks = nn.ModuleList(
            [
                ConvolutionalBlock(num_of_features*4, num_of_features*2, downsampling=False, kernel_size=3, stride=2, padding=1, output_padding=1),
                ConvolutionalBlock(num_of_features * 2, num_of_features * 1, downsampling=False, kernel_size=3, stride=2, padding=1, output_padding=1)
            ]
        )

        self.rgb = nn.Conv2d(num_of_features*1, image_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, x):
        x = self.init(x)
        for layer in self.downblocks:
            x = layer(x)
        x = self.residuals(x)
        for layer in self.upsample_blocks:
            x = layer(x)

        return torch.tanh(self.rgb(x))

# test case

def test():
    img_channels = 3
    img_size = 256
    x = torch.randn((2, img_channels, img_size, img_size))
    gen = Generator(img_channels, 9)
    print(gen)
    print(gen(x).shape)

if __name__ == "__main__":
    test()