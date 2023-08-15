import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch
import numpy as np
from model.convmixer import ConvMixer, ECA, ConvMix
from  model.S2ENet import Spatial_Enhance_Module, Spectral_Enhance_Module
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Residual(nn.Module):
    def __init__(self, dim, patch_size):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim, kernel_size=patch_size, groups=dim, padding=4)
        self.gelu = nn.GELU()
        self.bn = nn.BatchNorm3d(dim)

    def forward(self, x):
        # y = self.fn(x)
        y = self.conv(x)
        y = self.gelu(x)
        y = self.bn(x)

        return y + x


class conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding=0, dilation=1, groups=1, bias=True):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out


class PRTNet(nn.Module):
    def __init__(self, dim=1024, nhead=8, patch_size=7, depth=5, n_classes=2,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(PRTNet, self).__init__()
        a = [512, 256]
        # size = 9*11*9 #patch=5
        # size = 7*8*7 #patch=7
        # size = 5*6*5 #patch=9
        size = int(49//patch_size)*int(56//patch_size)*int(49//patch_size)


        # self.convMixer = ConvMixer(512, 3, 9, 4, 2)  # DCE_nums 0.63
        # self.convMixer = ConvMixer(512, 3, 9, 4, 2)  # T2_nums 0.64
        # self.conMixer = nn.Sequential(
        #     nn.Conv3d(1, dim, kernel_size=patch_size, stride=patch_size),
        #     nn.GELU(),
        #     nn.BatchNorm3d(dim),
        #     *[nn.Sequential(
        #         Residual(nn.Sequential(
        #             nn.Conv3d(dim, dim, kernel_size=patch_size, groups=dim, padding=4),
        #             nn.GELU(),
        #             nn.BatchNorm3d(dim)
        #         )),
        #         nn.Conv3d(dim, dim, kernel_size=1),
        #         nn.GELU(),
        #         nn.BatchNorm3d(dim)
        #     ) for i in range(depth)],
        #     nn.AdaptiveAvgPool3d((1, 1, 1)),
        #     nn.Flatten(),
        #     nn.Linear(dim, n_classes)
        # )
        self.con1 = conv_bn_relu(1, dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.con2 = conv_bn_relu(1, dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.convmix1 = ConvMix(dim, dim, depth=depth, kernel_size=patch_size, patch_size=patch_size, n_classes=2)
        self.convmix2 = ConvMix(dim, dim, depth=depth, kernel_size=patch_size, patch_size=patch_size, n_classes=2)
        self.conv11 = conv_bn_relu(dim, a[0], kernel_size=1, stride=2, bias=True)
        self.conv22 = conv_bn_relu(dim, a[0], kernel_size=1, stride=2, bias=True)
        self.SAEM = Spatial_Enhance_Module(in_channels=a[0], inter_channels=a[1], size=size)
        self.SEEM = Spectral_Enhance_Module(in_channels=a[0], in_channels2=a[0])
        self.FusionLayer = nn.Sequential(
            nn.Conv3d(in_channels=a[0] * 2, out_channels=a[0], kernel_size=1),
            nn.BatchNorm3d(a[0]),
            nn.ReLU(),
        )

        self.avgp = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flat = nn.Flatten()
        self.Linear = nn.Linear(a[0], n_classes)
        # self.linear1 = nn.Linear(dim, 64)
        # self.linear2 = nn.Linear(64, n_classes)
        self.sigmoid = nn.Sigmoid()

        # self.eca = ECA(d_model, nhead, dim_feedforward, dropout, activation)
        # self.FC = nn.Linear(1000, 2)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):

        # x = x.unsqueeze(0)
        # x = x.permute(1,0,2,3,4)
        # y = self.convMixer(x)
        x1 = self.con1(x1)
        x2 = self.con2(x2)
        x1 = self.convmix1(x1)
        x2 = self.convmix2(x2)
        # x2 = self.convmix1(x2)
        x1 = self.conv11(x1)
        x2 = self.conv22(x2)
        ss_x1 = self.SAEM(x1, x2)
        ss_x2 = self.SEEM(x1, x2)
        y = self.FusionLayer(torch.cat([ss_x1, ss_x2], 1))
        y = self.avgp(y)
        x = self.flat(y)
        y = self.Linear(x)
        # y = F.relu(self.linear1(y))
        # y = self.linear2(y)
        # x = self.sigmoid(x)
        # y = self.sigmoid(y)

        return x, y


if __name__ == '__main__':
    net = PRTNet()
    # for layer in PRTNet.modules():
    #     print(layer.weight)
    #     init.xavier_uniform(layer.weight)
    #     print(layer.weight)
