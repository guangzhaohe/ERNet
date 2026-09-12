"""Fixed ConvONet U-Net (jaxony/unet-pytorch); see THIRD_PARTY.md."""
import torch
from torch import nn
from torch.nn import functional as F


class DownConv(nn.Module):
    def __init__(self, channels_in, channels_out, pooling):
        super().__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, 3, padding=1)
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, padding=1)
        self.pool = nn.MaxPool2d(2) if pooling else nn.Identity()

    def forward(self, x):
        before_pool = F.relu(self.conv2(F.relu(self.conv1(x))))
        return self.pool(before_pool), before_pool


class UpConv(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(channels_in, channels_out, 2, stride=2)
        self.conv1 = nn.Conv2d(2 * channels_out, channels_out, 3, padding=1)
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, padding=1)

    def forward(self, skip, x):
        x = torch.cat((self.upconv(x), skip), 1)
        return F.relu(self.conv2(F.relu(self.conv1(x))))


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_convs = nn.ModuleList([
            DownConv(128 if i == 0 else 32 * 2 ** (i - 1), 32 * 2 ** i, i < 4)
            for i in range(5)
        ])
        self.up_convs = nn.ModuleList([
            UpConv(512 // 2 ** i, 256 // 2 ** i) for i in range(4)
        ])
        self.conv_final = nn.Conv2d(32, 128, 1)

    def forward(self, x):
        skips = []
        for down in self.down_convs:
            x, skip = down(x)
            skips.append(skip)
        for i, up in enumerate(self.up_convs):
            x = up(skips[-i - 2], x)
        return self.conv_final(x)
