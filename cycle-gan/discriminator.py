# Imports
import torch
import torch.nn as nn

# Block of Layers: Conv-InstanceNorm-ReLU
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride,
                      1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.conv(x)

# Discriminator Model : Initial conv + Block
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, block_features=[64, 128, 256, 512]):
        super().__init__()
        self.init = nn.Sequential(
            nn.Conv2d(
                in_channels,
                block_features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect"
            ),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_channels = block_features[0]

        for feature in block_features[1:]:
            layers.append((Block(in_channels, feature, stride=1 if feature==block_features[-1] else 2)))
            in_channels = feature
        # For mapping to 0 or 1, which indicates real or fake image.
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.init(x)
        return torch.sigmoid(self.model(x))

def test():
    x = torch.randn((5, 3, 256, 256)) # 5 examples, rgb, 256x256
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(model)
    print(preds.shape) # 30x30. Each value in this grid sees
    # 70x70 patch of the original image.(PatchGAN)


def test():
    x = torch.randn((5, 3, 256, 256)) # 5 examples, rgb, 256x256
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(model)
    print(preds.shape) # 30x30. Each value in this grid sees
    # 70x70 patch of the original image.(PatchGAN)


if __name__ == "__main__":
    test()

