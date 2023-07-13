import torch
import torch.nn as nn
import torch.nn.functional as F


class double_res_conv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=False):
        super(double_res_conv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
        )

        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.relu(x2)

        return x3


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True):
        super(inconv, self).__init__()

        self.conv = double_res_conv(in_ch, out_ch, bn)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.AvgPool2d(2), double_res_conv(in_ch, out_ch, bn))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, bn=True):
        super(up, self).__init__()

        self.bilinear = bilinear
        if not bilinear:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_res_conv(in_ch, out_ch, bn)

    def forward(self, x1, x2):
        if not self.bilinear:
            x1 = self.up(x1)
        else:
            x1 = nn.functional.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, padding=0)

    def forward(self, x):
        x = self.conv(x)

        return x


class PostUNet(nn.Module):
    def __init__(self, n_channels=1, scale=1):
        super(PostUNet, self).__init__()

        self.inc = inconv(n_channels, 64 // scale)
        self.down1 = down(64 // scale, 128 // scale)
        self.down2 = down(128 // scale, 256 // scale)
        self.down3 = down(256 // scale, 512 // scale)
        self.down4 = down(512 // scale, 512 // scale)

        self.up1 = up(1024 // scale, 256 // scale)
        self.up2 = up(512 // scale, 128 // scale)
        self.up3 = up(256 // scale, 64 // scale)
        self.up4 = up(128 // scale, 32 // scale)

        self.reduce = outconv(32 // scale, 1)

    def forward(self, x0):
        # print(x0.shape)
        x1 = self.inc(x0)
        # print(x1.shape)
        x2 = self.down1(x1)
        # print(x2.shape)
        x3 = self.down2(x2)
        # print(x3.shape)
        x4 = self.down3(x3)
        # print(x4.shape)
        x5 = self.down4(x4)
        # print(x5.shape)
        x = self.up1(x5, x4)
        # print(x.shape)
        x = self.up2(x, x3)
        # print(x.shape)
        x = self.up3(x, x2)
        # print(x.shape)
        x = self.up4(x, x1)
        # print(x.shape)
        x = self.reduce(x)
        # print(x.shape)
        x = x[:, 0, :, :]
        # print(x.shape)
        return x
