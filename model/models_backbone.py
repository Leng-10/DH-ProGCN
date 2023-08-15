import torch
import torch.nn as nn
from model.convmixer import ConvMixer, ECA, ConvMix
from model.S2ENet import Spatial_Enhance_Module, Spectral_Enhance_Module
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ConvMix_1：原始convmix dim1024 patch7 depth5
# ConvMix_MRF_1：convmix中的模块换成多尺度模块MRF
# ConvMix_MRF_CAG_1：convmix中的模块换成多尺度注意力模块MRF+CAG
# ConvMix_SWE_CWE_2：双模态convmix后接SWE+CWE
# ConvMix_MRF_CAG_SWE_CWE_2：双模态convmix模块替换成MRF+CAG后接SWE+CWE
__all__ = [
    'ConvMix_1',
    'Convmix_256', # for without_skull_image
    'Net_v2',
    'Net_v3',
    'ConvMix_MRF_1',
    'ConvMix_MRF_CAG_1',
    'ConvMix_SWE_CWE_2',
    'ConvMix_MRF_CAG_SWE_CWE_2',
]


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        y = self.fn(x)
        return y + x
# class Residual(nn.Module):
#     def __init__(self, dim, patch_size):
#         super().__init__()
#         self.conv = nn.Conv3d(dim, dim, kernel_size=patch_size, groups=dim, padding=4)
#         self.gelu = nn.GELU()
#         self.bn = nn.BatchNorm3d(dim)
#
#     def forward(self, x):
#         # y = self.fn(x)
#         y = self.conv(x)
#         y = self.gelu(x)
#         y = self.bn(x)
#
#         return y + x

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


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

class PSModule(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(PSModule, self).__init__()
        self.conv_1 = conv(inplans, planes//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                            stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                            stride=stride, groups=conv_groups[3])
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        out= torch.cat((x1, x2, x3, x4), dim=1)
        return out


def ConvMix_PS_Block(dim1,dim2, depth, conv_kernels):

    return nn.Sequential(
        *[nn.Sequential(
                Residual(nn.Sequential(
                    PSModule(dim1, dim2, conv_kernels, conv_groups=[dim2//4, dim2//4, dim2//4, dim2//4]),
                    nn.GELU(),
                    nn.BatchNorm3d(dim2)
                )),
                nn.Conv3d(dim2, dim2, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm3d(dim2)
        ) for i in range(depth)]
    )

# def ConvMix_PS_Block(dim1, dim2, depth, conv_kernels=[3, 5, 7, 9]):
#
#     return nn.Sequential(
#         *[nn.Sequential(
#                 Residual(nn.Sequential(
#                     nn.Conv3d(dim1, dim2, kernel_size=3, groups=dim2, padding="same"),
#                     # PSModule(dim1, dim2, conv_kernels=conv_kernels, conv_groups=[dim2//4, dim2//4, dim2//4, dim2//4]),
#                     nn.GELU(),
#                     nn.BatchNorm3d(dim2)
#                 )),
#                 nn.Conv3d(dim2, dim2, kernel_size=1),
#                 nn.GELU(),
#                 nn.BatchNorm3d(dim2)
#         ) for i in range(depth)]
#     )


class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight

class PSAModule(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(PSAModule, self).__init__()
        self.conv_1 = conv(inplans, planes//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                            stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                            stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3], feats.shape[4])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out

def ConvMix_PSA_Block(dim1, dim2, depth, conv_kernels=[3, 5, 7, 9]):

    return nn.Sequential(
        *[nn.Sequential(
                Residual(nn.Sequential(
                    # nn.Conv3d(dim1, dim2, kernel_size, groups=dim2, padding="same"),
                    PSAModule(dim1, dim2, conv_kernels=conv_kernels, conv_groups=[dim2//4, dim2//4, dim2//4, dim2//4]),
                    nn.GELU(),
                    nn.BatchNorm3d(dim2)
                )),
                nn.Conv3d(dim2, dim2, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm3d(dim2)
        ) for i in range(depth)]
    )

def ConvMix_v2(dim1, dim2, depth, kernel_size, patch_size=5, n_classes=2):

    return nn.Sequential(
        *[nn.Sequential(

                Residual(nn.Sequential(
                    # nn.Conv3d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.Conv3d(dim1, dim2, kernel_size, groups=dim2, padding="same"),
                    # nn.Conv3d(dim, dim, kernel_size, groups=dim),
                    nn.GELU(),
                    nn.BatchNorm3d(dim2)
                )),
                Residual(nn.Sequential(
                    nn.Conv3d(dim2, dim2, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm3d(dim2)
                ))

        ) for i in range(depth)]
    )





class ConvMix_1(nn.Module):
    def __init__(self, dim=1024, nhead=8, patch_size=7, depth=5, n_classes=2,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(ConvMix_1, self).__init__()
        a = [dim//2, dim//4]

        self.con1 = conv_bn_relu(1, dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.convmix1 = ConvMix(dim, dim, depth=depth, kernel_size=patch_size, patch_size=patch_size, n_classes=2)
        self.conv11 = conv_bn_relu(dim, a[0], kernel_size=1, stride=2, bias=True)
        self.conv22 = conv_bn_relu(a[0], a[0], kernel_size=1, stride=1)
        self.avgp = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flat = nn.Flatten()
        self.Linear = nn.Linear(a[0], n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1):

        x1 = self.con1(x1)
        x1 = self.convmix1(x1)
        x1 = self.conv11(x1)
        y = self.conv22(x1)
        y = self.avgp(y)
        x = self.flat(y)
        y = self.Linear(x)

        return x, y


class Convmix_256(nn.Module):
    def __init__(self, dim=1024, nhead=8, patch_size=8, kernel_size=7, depth=5, n_classes=2,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(Convmix_256, self).__init__()
        a = [dim, 512, 64]

        self.con1 = conv_bn_relu(1, a[0], kernel_size=patch_size, stride=patch_size, bias=True)
        self.convmix1 = ConvMix_v2(a[0], a[0], depth=depth, kernel_size=kernel_size, n_classes=2)
        self.conv11 = conv_bn_relu(a[0], a[1], kernel_size=1, stride=2, bias=True)
        # self.conv22 = conv_bn_relu(a[0], a[0], kernel_size=1, stride=1)

        self.avgp = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flat = nn.Flatten()
        # self.Linear = nn.Linear(a[0], n_classes)
        self.linear1 = nn.Linear(a[1], a[2])
        self.linear2 = nn.Linear(a[2], n_classes)

        self.dropout = nn.Dropout(p=0.3)
        self.dropout3d = nn.Dropout3d(p=0.1, inplace=False)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.con1(x)
        x = self.convmix1(x)
        x_convmix = self.conv11(x)
        # y = self.conv22(x1)

        x = self.avgp(x_convmix)
        y_64 = self.flat(x)
        y = self.linear2(self.linear1(y_64))

        return y_64, y
        # return y





class Net_v2(nn.Module):
    def __init__(self, dim=1024, nhead=8, patch_size=7, depth=5, n_classes=2,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(Net_v2, self).__init__()
        a = [dim//2, dim//4]
        # size = int(49//patch_size)*int(56//patch_size)*int(49//patch_size)

        self.con1 = conv_bn_relu(1, dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.convmix1 = ConvMix_v2(dim, dim, depth=depth, kernel_size=patch_size, n_classes=2)
        self.conv11 = conv_bn_relu(dim, a[0], kernel_size=1, stride=2, bias=True)
        self.conv22 = conv_bn_relu(a[0], a[0], kernel_size=1, stride=1)

        self.avgp = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flat = nn.Flatten()
        self.Linear = nn.Linear(a[0], n_classes)
        # self.linear1 = nn.Linear(dim, 64)
        # self.linear2 = nn.Linear(64, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1):

        x1 = self.con1(x1)
        x1 = self.convmix1(x1)
        x1 = self.conv11(x1)
        y = self.conv22(x1)

        y = self.avgp(y)
        x = self.flat(y)
        y = self.Linear(x)

        return x, y


class Net_v3(nn.Module):
    def __init__(self, dim=1024, nhead=8, patch_size=7, depth=5, n_classes=2,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(Net_v3, self).__init__()
        a = [512, 256]
        # size = 9*11*9 #patch=5
        # size = 7*8*7 #patch=7
        # size = 5*6*5 #patch=9
        size = int(49//patch_size)*int(56//patch_size)*int(49//patch_size)

        self.con1 = conv_bn_relu(1, dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.con2 = conv_bn_relu(1, dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.convmix1 = ConvMix_v2(dim, dim, depth=depth, kernel_size=patch_size, patch_size=patch_size, n_classes=2)
        self.convmix2 = ConvMix_v2(dim, dim, depth=depth, kernel_size=patch_size, patch_size=patch_size, n_classes=2)
        self.conv11 = conv_bn_relu(dim, a[0], kernel_size=1, stride=2, bias=True)
        self.conv22 = conv_bn_relu(a[0], a[0], kernel_size=1, stride=1)
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
        self.dropout = nn.Dropout(p=0.3)
        self.dropout3d = nn.Dropout3d(p=0.1, inplace=False)

        # self.eca = ECA(d_model, nhead, dim_feedforward, dropout, activation)
        # self.FC = nn.Linear(1000, 2)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1):

        # x = x.unsqueeze(0)
        # x = x.permute(1,0,2,3,4)
        # y = self.convMixer(x)
        x1 = self.con1(x1)
        x1 = self.convmix1(x1)
        x1 = self.conv11(x1)
        y = self.conv22(x1)
        # ss_x1 = self.SAEM(x1, x2)
        # ss_x2 = self.SEEM(x1, x2)
        # y = self.FusionLayer(torch.cat([ss_x1, ss_x2], 1))
        y = self.avgp(y)
        x = self.flat(y)
        y = self.Linear(x)
        # y = F.relu(self.linear1(y))
        # y = self.linear2(y)
        # x = self.sigmoid(x)
        # y = self.sigmoid(y)

        # return x, y
        return y












class ConvMix_MRF_1(nn.Module):
    def __init__(self, dim=1024, patch_size=5, depth=5, convs_k=[3, 5, 7, 9], n_classes=2,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(ConvMix_MRF_1, self).__init__()
        a = [dim//2, dim//4]

        self.con1 = conv_bn_relu(1, dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.MRF = ConvMix_PS_Block(dim, dim, depth=depth, conv_kernels=convs_k)
        self.conv11 = conv_bn_relu(dim, a[0], kernel_size=1, stride=1, bias=False)
        self.avgp = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flat = nn.Flatten()
        self.Linear = nn.Linear(a[0], n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x1):

        x1 = self.con1(x1)
        x1 = self.MRF(x1)
        x1 = self.conv11(x1)
        # x1 = self.conv22(x1)
        y = self.avgp(x1)
        x = self.flat(y)
        y = self.Linear(x)
        # y = F.relu(self.linear1(y))

        return x, y


class ConvMix_MRF_CAG_1(nn.Module):
    def __init__(self, dim=1024, nhead=8, patch_size=5, depth=5, convs_k=[3, 5, 7, 9], n_classes=2,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(ConvMix_MRF_CAG_1, self).__init__()
        a = [dim//2, dim//4]

        self.con1 = conv_bn_relu(1, dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.MRF = ConvMix_PSA_Block(dim, dim, depth=depth, conv_kernels=convs_k)
        self.conv11 = conv_bn_relu(dim, a[0], kernel_size=1, stride=1, bias=False)
        self.avgp = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flat = nn.Flatten()
        self.Linear = nn.Linear(a[0], n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x1):

        x1 = self.con1(x1)
        x1 = self.MRF(x1)
        x1 = self.conv11(x1)
        # x1 = self.conv22(x1)
        y = self.avgp(x1)
        x = self.flat(y)
        y = self.Linear(x)
        # y = F.relu(self.linear1(y))

        return x, y


class ConvMix_2(nn.Module):
    def __init__(self, dim=1024, nhead=8, patch_size=7, depth=5, n_classes=2,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(ConvMix_2, self).__init__()
        a = [dim//2, dim//4]

        self.con1 = conv_bn_relu(1, dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.con2 = conv_bn_relu(1, dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.convmix1 = ConvMix(dim, dim, depth=depth, kernel_size=patch_size, patch_size=patch_size, n_classes=2)
        self.convmix2 = ConvMix(dim, dim, depth=depth, kernel_size=patch_size, patch_size=patch_size, n_classes=2)
        self.conv11 = conv_bn_relu(dim, a[0], kernel_size=1, stride=2, bias=True)
        self.conv22 = conv_bn_relu(dim, a[0], kernel_size=1, stride=2, bias=True)
        # self.SAEM = Spatial_Enhance_Module(in_channels=a[0], inter_channels=a[1], size=size)
        # self.SEEM = Spectral_Enhance_Module(in_channels=a[0], in_channels2=a[0])
        # self.FusionLayer = nn.Sequential(
        #     nn.Conv3d(in_channels=a[0] * 2, out_channels=a[0], kernel_size=1),
        #     nn.BatchNorm3d(a[0]),
        #     nn.ReLU(),
        # )

        self.avgp = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flat = nn.Flatten()
        self.Linear = nn.Linear(1024, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x1=x
        x2=x
        x1 = self.con1(x1)
        x2 = self.con2(x2)
        x1 = self.convmix1(x1)
        x2 = self.convmix2(x2)
        x1 = self.conv11(x1)
        x2 = self.conv22(x2)
        y = torch.cat([x1, x2], 1)
        
        # ss_x1 = self.SAEM(x1, x2)
        # ss_x2 = self.SEEM(x1, x2)
        # y = self.FusionLayer(torch.cat([ss_x1, ss_x2], 1))
        
        y = self.avgp(y)
        x = self.flat(y)
        y = self.Linear(x)

        return x, y


class ConvMix_MRF_2(nn.Module):
    def __init__(self, dim=1024, nhead=8, patch_size=7, depth=5, conv_kernels=[3, 5, 7, 9], n_classes=2,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(ConvMix_MRF_2, self).__init__()
        a = [dim//2, dim//4]

        self.con1 = conv_bn_relu(1, dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.con2 = conv_bn_relu(1, dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.convmix1 = ConvMix_PS_Block(dim, dim, depth=depth, conv_kernels=conv_kernels)
        self.convmix2 = ConvMix_PS_Block(dim, dim, depth=depth, conv_kernels=conv_kernels)
        self.conv11 = conv_bn_relu(dim, a[0], kernel_size=1, stride=2, bias=True)
        self.conv22 = conv_bn_relu(dim, a[0], kernel_size=1, stride=2, bias=True)

        self.avgp = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flat = nn.Flatten()
        self.Linear = nn.Linear(1024, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = x
        x2 = x
        x1 = self.con1(x1)
        x2 = self.con2(x2)
        x1 = self.convmix1(x1)
        x2 = self.convmix2(x2)
        x1 = self.conv11(x1)
        x2 = self.conv22(x2)
        y = torch.cat([x1, x2], 1)

        # ss_x1 = self.SAEM(x1, x2)
        # ss_x2 = self.SEEM(x1, x2)
        # y = self.FusionLayer(torch.cat([ss_x1, ss_x2], 1))

        y = self.avgp(y)
        x = self.flat(y)
        y = self.Linear(x)

        return x, y


class ConvMix_MRF_CAG_2(nn.Module):
    def __init__(self, dim=1024, nhead=8, conv_kernels=[3, 5, 7, 9], patch_size=7, depth=5, n_classes=2,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(ConvMix_MRF_CAG_2, self).__init__()
        a = [dim//2, dim//4]
        size = int(49 // patch_size) * int(56 // patch_size) * int(49 // patch_size)

        self.con1 = conv_bn_relu(1, dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.con2 = conv_bn_relu(1, dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.convmix1 = ConvMix_PSA_Block(dim, dim, conv_kernels=conv_kernels, depth=depth)
        self.convmix2 = ConvMix_PSA_Block(dim, dim, conv_kernels=conv_kernels, depth=depth)
        self.conv11 = conv_bn_relu(dim, a[0], kernel_size=1, stride=2, bias=True)
        self.conv22 = conv_bn_relu(dim, a[0], kernel_size=1, stride=2, bias=True)
        # self.SAEM = Spatial_Enhance_Module(in_channels=a[0], inter_channels=a[1], size=size)
        # self.SEEM = Spectral_Enhance_Module(in_channels=a[0], in_channels2=a[0])
        # self.FusionLayer = nn.Sequential(
        #     nn.Conv3d(in_channels=a[0] * 2, out_channels=a[0], kernel_size=1),
        #     nn.BatchNorm3d(a[0]),
        #     nn.ReLU(),
        # )

        self.avgp = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flat = nn.Flatten()
        self.Linear = nn.Linear(1024, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x1=x
        x2=x
        x1 = self.con1(x1)
        x2 = self.con2(x2)
        x1 = self.convmix1(x1)
        x2 = self.convmix2(x2)
        x1 = self.conv11(x1)
        x2 = self.conv22(x2)
        y = torch.cat([x1, x2], 1)

        # ss_x1 = self.SAEM(x1, x2)
        # ss_x2 = self.SEEM(x1, x2)
        # y = self.FusionLayer(torch.cat([ss_x1, ss_x2], 1))

        y = self.avgp(y)
        x = self.flat(y)
        y = self.Linear(x)

        return x, y


class ConvMix_SWE_CWE_2(nn.Module):
    def __init__(self, dim=1024, nhead=8, patch_size=7, depth=5, n_classes=2,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(ConvMix_SWE_CWE_2, self).__init__()
        a = [dim//2, dim//4]
        size = int(49 // patch_size) * int(56 // patch_size) * int(49 // patch_size)

        self.con1 = conv_bn_relu(1, dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.con2 = conv_bn_relu(1, dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.convmix1 = ConvMix(dim, dim, depth=depth, kernel_size=patch_size, patch_size=patch_size, n_classes=2)
        self.convmix2 = ConvMix(dim, dim, depth=depth, kernel_size=patch_size, patch_size=patch_size, n_classes=2)
        # self.convmix1 = ConvMix_v2(dim, dim, depth=depth, kernel_size=patch_size, patch_size=patch_size, n_classes=2)
        # self.convmix2 = ConvMix_v2(dim, dim, depth=depth, kernel_size=patch_size, patch_size=patch_size, n_classes=2)
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

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y):

        x1=x
        x2=y
        x1 = self.con1(x1)
        x2 = self.con2(x2)
        x1 = self.convmix1(x1)
        x2 = self.convmix2(x2)
        x1 = self.conv11(x1)
        x2 = self.conv22(x2)
        # y = torch.cat([x1, x2], 1)

        ss_x1 = self.SAEM(x1, x2)
        ss_x2 = self.SEEM(x1, x2)
        y = self.FusionLayer(torch.cat([ss_x1, ss_x2], 1))

        y = self.avgp(y)
        x = self.flat(y)
        y = self.Linear(x)

        return x, y



class ConvMix_MRF_CAG_SWE_CWE_2(nn.Module):
    def __init__(self, dim=1024, nhead=8, conv_kernels=[3, 5, 7, 9], patch_size=7, depth=5, n_classes=2,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(ConvMix_MRF_CAG_SWE_CWE_2, self).__init__()
        a = [dim//2, dim//4]
        size = int(49 // patch_size) * int(56 // patch_size) * int(49 // patch_size)

        self.con1 = conv_bn_relu(1, dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.con2 = conv_bn_relu(1, dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.convmix1 = ConvMix_PSA_Block(dim, dim, conv_kernels=conv_kernels, depth=depth)
        self.convmix2 = ConvMix_PSA_Block(dim, dim, conv_kernels=conv_kernels, depth=depth)
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

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x1=x
        x2=x
        x1 = self.con1(x1)
        x2 = self.con2(x2)
        x1 = self.convmix1(x1)
        x2 = self.convmix2(x2)
        x1 = self.conv11(x1)
        x2 = self.conv22(x2)
        # y = torch.cat([x1, x2], 1)

        ss_x1 = self.SAEM(x1, x2)
        ss_x2 = self.SEEM(x1, x2)
        y = self.FusionLayer(torch.cat([ss_x1, ss_x2], 1))

        y = self.avgp(y)
        x = self.flat(y)
        y = self.Linear(x)

        return x, y









class ConvMix_2(nn.Module):
    def __init__(self, dim=1024, nhead=8, patch_size=7, depth=5, n_classes=2,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(ConvMix_2, self).__init__()
        a = [dim//2, dim//4]
        size = int(49//patch_size)*int(56//patch_size)*int(49//patch_size)

        self.con1 = conv_bn_relu(1, dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.con2 = conv_bn_relu(1, dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.convmix1 = ConvMix(dim, dim, depth=depth, kernel_size=patch_size, patch_size=patch_size, n_classes=2)
        self.convmix2 = ConvMix(dim, dim, depth=depth, kernel_size=patch_size, patch_size=patch_size, n_classes=2)
        self.conv11 = conv_bn_relu(dim, a[0], kernel_size=1, stride=2, bias=True)
        self.conv22 = conv_bn_relu(a[0], a[0], kernel_size=1, stride=1)
        self.SAEM = Spatial_Enhance_Module(in_channels=a[0], inter_channels=a[1], size=size)
        # self.SEEM = Spectral_Enhance_Module(in_channels=a[0], in_channels2=a[0])
        # self.FusionLayer = nn.Sequential(
        #     nn.Conv3d(in_channels=a[0] * 2, out_channels=a[0], kernel_size=1),
        #     nn.BatchNorm3d(a[0]),
        #     nn.ReLU(),
        # )

        self.avgp = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flat = nn.Flatten()
        self.Linear = nn.Linear(a[0], n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1):

        x1 = self.con1(x1)
        x1 = self.convmix1(x1)
        x1 = self.conv11(x1)
        y = self.conv22(x1)
        # ss_x1 = self.SAEM(x1, x2)
        # ss_x2 = self.SEEM(x1, x2)
        # y = self.FusionLayer(torch.cat([ss_x1, ss_x2], 1))
        y = self.avgp(y)
        x = self.flat(y)
        y = self.Linear(x)


        return x, y



class PRTNet(nn.Module):
    def __init__(self, dim=1024, nhead=8, patch_size=7, depth=5, n_classes=2,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(PRTNet, self).__init__()
        a = [dim//2, dim//4]
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
        self.conv22 = conv_bn_relu(a[0], a[0], kernel_size=1, stride=1)
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
        self.dropout = nn.Dropout(p=0.3)

        # self.eca = ECA(d_model, nhead, dim_feedforward, dropout, activation)
        # self.FC = nn.Linear(1000, 2)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1):

        # x = x.unsqueeze(0)
        # x = x.permute(1,0,2,3,4)
        # y = self.convMixer(x)
        x1 = self.con1(x1)
        x1 = self.convmix1(x1)
        x1 = self.conv11(x1)
        y = self.conv22(x1)
        # ss_x1 = self.SAEM(x1, x2)
        # ss_x2 = self.SEEM(x1, x2)
        # y = self.FusionLayer(torch.cat([ss_x1, ss_x2], 1))
        y = self.avgp(y)
        x = self.flat(y)
        y = self.Linear(x)
        # y = F.relu(self.linear1(y))
        # y = self.linear2(y)
        # x = self.sigmoid(x)
        # y = self.sigmoid(y)

        return x, y

