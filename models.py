import torch
import torchvision

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import sys
import math


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class ActivationFunc(nn.Module):
    def __init__(self, func_type):
        super(ActivationFunc, self).__init__()
        if func_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif func_type == 'leaky':
            self.act = nn.LeakyReLU(inplace=True)
        elif func_type == 'mish':
            self.act = Mish()
        elif func_type == 'linear' or func_type == 'none':
            self.act = nn.Identity()
        else:
            print("Invalid activation! {} {} {}".format(sys._getframe().f_code.co_filename,
                                                        sys._getframe().f_code.co_name, sys._getframe().f_lineno))

    def forward(self, x):
        return self.act(x)


class DenseLayer(nn.Module):
    def __init__(self, n_channels, growth_rate, activation='relu', dw_sep_conv=False):
        """

        :param n_channels: Number of input channles
        :param growth_rate: Number of output channels of single layer
        :param dw_sep_conv: Replace regular 2D conv with Depthwise-Separable Conv
        """
        super(DenseLayer, self).__init__()

        self.bn1 = nn.BatchNorm2d(n_channels)

        self.act = ActivationFunc(activation)

        if dw_sep_conv:
            self.conv1 = nn.Sequential(
                nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False, groups=n_channels),
                nn.BatchNorm2d(n_channels),
                ActivationFunc(activation),
                nn.Conv2d(n_channels, growth_rate, kernel_size=1, padding=1, bias=False)
            )
        else:
            self.conv1 = nn.Conv2d(n_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.act(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class TransitionLayer(nn.Module):
    def __init__(self, n_channels, n_out_channels, activation='relu', dw_sep_conv=False):
        """

        :param n_channels: Number of input channles
        :param n_out_channels: Number of output channels of single layer
        :param dw_sep_conv: Replace regular 2D conv with Depthwise-Separable Conv
        """
        super(TransitionLayer, self).__init__()

        self.bn1 = nn.BatchNorm2d(n_channels)

        self.act = ActivationFunc(activation)

        if dw_sep_conv:
            self.conv1 = nn.Sequential(
                nn.Conv2d(n_channels, n_channels, kernel_size=1, padding=1, bias=False, groups=n_channels),
                nn.BatchNorm2d(n_channels),
                ActivationFunc(activation),
                nn.Conv2d(n_channels, n_out_channels, kernel_size=1, padding=1, bias=False)
            )
        else:
            self.conv1 = nn.Conv2d(n_channels, n_out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.act(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class PartialTransitionLayer(nn.Module):
    def __init__(self, n_channels, n_out_channels, activation='relu', dw_sep_conv=False):
        """

        :param n_channels: Number of input channles
        :param n_out_channels: Number of output channels of single layer
        :param dw_sep_conv: Replace regular 2D conv with Depthwise-Separable Conv
        """
        super(PartialTransitionLayer, self).__init__()

        self.bn1 = nn.BatchNorm2d(n_channels)

        self.act = ActivationFunc(activation)

        if dw_sep_conv:
            self.conv1 = nn.Sequential(
                nn.Conv2d(n_channels, n_channels, kernel_size=1, padding=1, bias=False, groups=n_channels),
                nn.BatchNorm2d(n_channels),
                ActivationFunc(activation),
                nn.Conv2d(n_channels, n_out_channels, kernel_size=1, padding=1, bias=False)
            )
        else:
            self.conv1 = nn.Conv2d(n_channels, n_out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.act(self.bn1(x)))
        return out


class BottleneckLayer(nn.Module):
    def __init__(self, n_channels, growth_rate, activation='relu', dw_sep_conv=False):
        """

        :param n_channels: Number of input channles
        :param growth_rate: Number of output channels of single layer
        :param dw_sep_conv: Replace regular 2D conv with Depthwise-Separable Conv
        """
        super(BottleneckLayer, self).__init__()
        inter_channels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.bn2 = nn.BatchNorm2d(inter_channels)

        self.act1 = ActivationFunc(activation)
        self.act2 = ActivationFunc(activation)

        if dw_sep_conv:
            self.conv1 = nn.Sequential(
                nn.Conv2d(n_channels, n_channels, kernel_size=1, bias=False, groups=n_channels),
                nn.BatchNorm2d(n_channels),
                ActivationFunc(activation),
                nn.Conv2d(n_channels, inter_channels, kernel_size=1, bias=False)
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1, bias=False, groups=n_channels),
                nn.BatchNorm2d(n_channels),
                ActivationFunc(activation),
                nn.Conv2d(inter_channels, growth_rate, kernel_size=1, padding=1, bias=False)
            )
        else:
            self.conv1 = nn.Conv2d(n_channels, inter_channels, kernel_size=1, bias=False)
            self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.act1(self.bn1(x)))
        out = self.conv2(self.act2(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out


class DenseBlock(nn.Module):
    def __init__(self, n_channels, growth_rate, n_layers, activation='relu', bottleneck=False, dw_sep_conv=False):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(n_layers):
            if bottleneck:
                layers.append(BottleneckLayer(n_channels, growth_rate, activation, dw_sep_conv=dw_sep_conv))
            else:
                layers.append(DenseLayer(n_channels, growth_rate, activation, dw_sep_conv=dw_sep_conv))
            n_channels += growth_rate

        self.dense_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.dense_block(x)


class CSPDenseBlock(nn.Module):
    def __init__(self, n_channels, growth_rate, n_layers, activation, bottleneck=False, dw_sep_conv=False):
        """
        ToDo: Finish implementing CSPDenseBlock, publications unclear on how to define transition layers

        :param n_channels: Number of input channels
        :param growth_rate: Number of output channels of single layer
        :param dw_sep_conv: Replace regular 2D conv with Depthwise-Separable Conv
        """
        super(CSPDenseBlock, self).__init__()

        self.n_channels = n_channels
        self.growth_rate = growth_rate

        self.dense = DenseBlock(n_channels, growth_rate, n_layers, activation, bottleneck, dw_sep_conv)

        pn_channels = n_channels + (n_layers * growth_rate)
        # self.transition = PartialTransitionLayer(pn_channels, )

    def forward(self, x):
        x1, x2 = torch.split(x, self.n_channels // 2, dim=1)
        out = self.dense(x2)
        # out = self.transition(out)
        out = torch.cat((x1, out), 1)
        return out


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False, activation='relu',
                 dw_sep_conv=False):
        super(ConvBNAct, self).__init__()
        padding = (kernel_size - 1) // 2
        if dw_sep_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=bias),
                nn.BatchNorm2d(in_channels),
                ActivationFunc(activation),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                  bias=bias)

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = ActivationFunc(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        return out


class ResidualBlock(nn.Module):
    """
        Generic function to apply residual shortcuts. Just overwrite self.blocks
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.blocks = nn.Identity()

        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x

    @property
    def apply_shortcut(self):
        return self.in_channels != self.out_channels


class DarkResBlock(ResidualBlock):
    def __init__(self, channels, n_blocks=1, activation='relu', dw_sep_conv=False):
        super(DarkResBlock, self).__init__(in_channels=channels, out_channels=channels)
        layers = []
        for i in range(n_blocks):
            layers.append(
                ConvBNAct(channels, channels, kernel_size=1, stride=1, bias=False, activation=activation,
                          dw_sep_conv=False))
            layers.append(
                ConvBNAct(channels, channels, kernel_size=3, stride=1, bias=False, activation=activation,
                          dw_sep_conv=dw_sep_conv))

        self.blocks = nn.Sequential(*layers)


"""
    DARKNET53
"""


class DarkNet53(nn.Module):
    def __init__(self, num_classes=1000, activation='relu', dw_sep_conv=False):
        super(DarkNet53, self).__init__()

        self.conv1 = ConvBNAct(3, 32, kernel_size=3, stride=1, bias=False, activation=activation,
                               dw_sep_conv=dw_sep_conv)
        self.conv2 = ConvBNAct(32, 64, kernel_size=3, stride=2, bias=False, activation=activation,
                               dw_sep_conv=dw_sep_conv)

        # Block 1
        self.conv_b1_1 = ConvBNAct(64, 32, kernel_size=1, bias=False, activation=activation)
        self.conv_b1_2 = ConvBNAct(32, 64, kernel_size=3, stride=1, bias=False, activation=activation,
                                   dw_sep_conv=dw_sep_conv)

        # Reduction 1
        self.conv_red1 = ConvBNAct(64, 128, kernel_size=3, stride=2, bias=False, activation=activation,
                                   dw_sep_conv=dw_sep_conv)

        # Block 2
        self.conv_b2_1 = ConvBNAct(128, 64, kernel_size=1, bias=False, activation=activation)
        self.conv_b2_2 = ConvBNAct(64, 128, kernel_size=3, stride=1, bias=False, activation=activation,
                                   dw_sep_conv=dw_sep_conv)
        self.conv_b2_3 = ConvBNAct(128, 64, kernel_size=1, bias=False, activation=activation)
        self.conv_b2_4 = ConvBNAct(64, 128, kernel_size=3, stride=1, bias=False, activation=activation,
                                   dw_sep_conv=dw_sep_conv)

        # Reduction 2
        self.conv_red2 = ConvBNAct(128, 256, kernel_size=3, stride=2, bias=False, activation=activation,
                                   dw_sep_conv=dw_sep_conv)

        # Block 3
        self.conv_b3_1 = ConvBNAct(256, 128, kernel_size=1, bias=False, activation=activation)
        self.conv_b3_2 = ConvBNAct(128, 256, kernel_size=3, stride=1, bias=False, activation=activation,
                                   dw_sep_conv=dw_sep_conv)
        self.conv_b3_3 = ConvBNAct(256, 128, kernel_size=1, bias=False, activation=activation)
        self.conv_b3_4 = ConvBNAct(128, 256, kernel_size=3, stride=1, bias=False, activation=activation,
                                   dw_sep_conv=dw_sep_conv)
        self.conv_b3_5 = ConvBNAct(256, 128, kernel_size=1, bias=False, activation=activation)
        self.conv_b3_6 = ConvBNAct(128, 256, kernel_size=3, stride=1, bias=False, activation=activation,
                                   dw_sep_conv=dw_sep_conv)
        self.conv_b3_7 = ConvBNAct(256, 128, kernel_size=1, bias=False, activation=activation)
        self.conv_b3_8 = ConvBNAct(128, 256, kernel_size=3, stride=1, bias=False, activation=activation,
                                   dw_sep_conv=dw_sep_conv)

        # Reduction 3
        self.conv_red3 = ConvBNAct(256, 512, kernel_size=3, stride=2, bias=False, activation=activation,
                                   dw_sep_conv=dw_sep_conv)

        # Block 4
        self.conv_b4_1 = ConvBNAct(512, 256, kernel_size=1, bias=False, activation=activation)
        self.conv_b4_2 = ConvBNAct(256, 512, kernel_size=3, stride=1, bias=False, activation=activation,
                                   dw_sep_conv=dw_sep_conv)
        self.conv_b4_3 = ConvBNAct(512, 256, kernel_size=1, bias=False, activation=activation)
        self.conv_b4_4 = ConvBNAct(256, 512, kernel_size=3, stride=1, bias=False, activation=activation,
                                   dw_sep_conv=dw_sep_conv)
        self.conv_b4_5 = ConvBNAct(512, 256, kernel_size=1, bias=False, activation=activation)
        self.conv_b4_6 = ConvBNAct(256, 512, kernel_size=3, stride=1, bias=False, activation=activation,
                                   dw_sep_conv=dw_sep_conv)
        self.conv_b4_7 = ConvBNAct(512, 256, kernel_size=1, bias=False, activation=activation)
        self.conv_b4_8 = ConvBNAct(256, 512, kernel_size=3, stride=1, bias=False, activation=activation,
                                   dw_sep_conv=dw_sep_conv)

        # Reduction 4
        self.conv_red4 = ConvBNAct(512, 1024, kernel_size=3, stride=2, bias=False, activation=activation,
                                   dw_sep_conv=dw_sep_conv)

        self.conv_b5_1 = ConvBNAct(1024, 512, kernel_size=1, bias=False, activation=activation)
        self.conv_b5_2 = ConvBNAct(512, 1024, kernel_size=3, stride=1, bias=False, activation=activation,
                                   dw_sep_conv=dw_sep_conv)
        self.conv_b5_3 = ConvBNAct(1024, 512, kernel_size=1, bias=False, activation=activation)
        self.conv_b5_4 = ConvBNAct(512, 1024, kernel_size=3, stride=1, bias=False, activation=activation,
                                   dw_sep_conv=dw_sep_conv)

        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)

        # Block 1
        xb1 = self.conv_b1_1(x2)
        xb1 = self.conv_b1_2(xb1)
        x3 = xb1 + x2

        x3 = self.conv_red1(x3)

        # Block 2
        xb2 = self.conv_b2_1(x3)
        xb2 = self.conv_b2_2(xb2)
        xb2 = self.conv_b2_3(xb2)
        xb2 = self.conv_b2_4(xb2)
        x4 = xb2 + x3

        x4 = self.conv_red2(x4)

        # Block 3
        xb3 = self.conv_b3_1(x4)
        xb3 = self.conv_b3_2(xb3)
        xb3 = self.conv_b3_3(xb3)
        xb3 = self.conv_b3_4(xb3)
        xb3 = self.conv_b3_5(xb3)
        xb3 = self.conv_b3_6(xb3)
        xb3 = self.conv_b3_7(xb3)
        xb3 = self.conv_b3_8(xb3)
        x5 = xb3 + x4

        x5 = self.conv_red3(x5)

        # Block 4
        xb4 = self.conv_b4_1(x5)
        xb4 = self.conv_b4_2(xb4)
        xb4 = self.conv_b4_3(xb4)
        xb4 = self.conv_b4_4(xb4)
        xb4 = self.conv_b4_5(xb4)
        xb4 = self.conv_b4_6(xb4)
        xb4 = self.conv_b4_7(xb4)
        xb4 = self.conv_b4_8(xb4)
        x6 = xb4 + x5

        x6 = self.conv_red4(x6)

        # Block 5
        xb5 = self.conv_b5_1(x6)
        xb5 = self.conv_b5_2(xb5)
        xb5 = self.conv_b5_3(xb5)
        xb5 = self.conv_b5_4(xb5)
        x7 = xb5 + x6

        x7 = F.avg_pool2d(x7, 8)
        x7 = torch.squeeze(x7)
        x_fc = self.fc(x7)
        out = F.log_softmax(x_fc, dim=0)

        return out


"""
    CSPDARKNET53
"""


class CSPDarkNet53(nn.Module):
    def __init__(self, num_classes=1000, activation='relu', dw_sep_conv=False):
        """
        ToDo: put these darknet blocks into a separate generic class since the pattern in the same, only number of channels changed
        :param num_classes:
        :param activation:
        :param dw_sep_conv:
        """
        super(CSPDarkNet53, self).__init__()
        ''' Input Conv Layer '''
        self.conv1 = ConvBNAct(3, 32, kernel_size=3, stride=1, bias=False, activation=activation,
                               dw_sep_conv=dw_sep_conv)

        ''' Downsample 1 '''
        self.convd1_1 = ConvBNAct(32, 64, kernel_size=3, stride=2, activation=activation,
                                  dw_sep_conv=dw_sep_conv)

        self.convd1_2 = ConvBNAct(64, 64, kernel_size=1, stride=1, activation=activation,
                                  dw_sep_conv=False)
        # [route] layers=-2
        self.convd1_3 = ConvBNAct(64, 64, kernel_size=3, stride=1, activation=activation,
                                  dw_sep_conv=dw_sep_conv)
        self.convd1_4 = ConvBNAct(64, 32, kernel_size=3, stride=1, activation=activation,
                                  dw_sep_conv=dw_sep_conv)
        self.convd1_5 = ConvBNAct(32, 64, kernel_size=3, stride=1, bias=False, activation=activation,
                                  dw_sep_conv=dw_sep_conv)
        # [shortcut] from=-3
        self.convd1_6 = ConvBNAct(64, 64, kernel_size=1, stride=1, bias=False, activation=activation,
                                  dw_sep_conv=False)
        # [route] layers = -1, -7
        self.convd1_7 = ConvBNAct(128, 64, kernel_size=1, stride=1, bias=False, activation=activation,
                                  dw_sep_conv=False)

        '''Downsample 2'''
        self.convd2_1 = ConvBNAct(64, 128, kernel_size=3, stride=2, activation=activation,
                                  dw_sep_conv=dw_sep_conv)
        self.convd2_2 = ConvBNAct(128, 64, kernel_size=1, stride=1, activation=activation,
                                  dw_sep_conv=False)
        # [route] layers=-2
        self.convd2_3 = ConvBNAct(128, 64, kernel_size=1, stride=1, activation=activation,
                                  dw_sep_conv=dw_sep_conv)

        self.resblockd2 = DarkResBlock(64, n_blocks=2, activation=activation, dw_sep_conv=dw_sep_conv)

        # shortcut -3
        self.convd2_4 = ConvBNAct(64, 64, 1, 1, bias=False, activation=activation, dw_sep_conv=False)
        # route -1 -10
        self.convd2_5 = ConvBNAct(128, 128, 1, 1, bias=False, activation=activation, dw_sep_conv=False)

        '''Downsample 3'''
        self.convd3_1 = ConvBNAct(128, 256, kernel_size=3, stride=2, activation=activation,
                                  dw_sep_conv=dw_sep_conv)
        self.convd3_2 = ConvBNAct(256, 128, kernel_size=1, stride=1, activation=activation,
                                  dw_sep_conv=False)
        # [route] layers=-2
        self.convd3_3 = ConvBNAct(256, 128, kernel_size=1, stride=1, activation=activation,
                                  dw_sep_conv=dw_sep_conv)

        self.resblockd3 = DarkResBlock(128, n_blocks=8, activation=activation, dw_sep_conv=dw_sep_conv)

        # shortcut -3
        self.convd3_4 = ConvBNAct(128, 128, 1, 1, bias=False, activation=activation, dw_sep_conv=False)
        # route -1 -10
        self.convd3_5 = ConvBNAct(256, 256, 1, 1, bias=False, activation=activation, dw_sep_conv=False)

        '''Downsample 4'''
        self.convd4_1 = ConvBNAct(256, 512, kernel_size=3, stride=2, activation=activation,
                                  dw_sep_conv=dw_sep_conv)
        self.convd4_2 = ConvBNAct(512, 256, kernel_size=1, stride=1, activation=activation,
                                  dw_sep_conv=False)
        # [route] layers=-2
        self.convd4_3 = ConvBNAct(512, 256, kernel_size=1, stride=1, activation=activation,
                                  dw_sep_conv=dw_sep_conv)

        self.resblockd4 = DarkResBlock(256, n_blocks=8, activation=activation, dw_sep_conv=dw_sep_conv)

        # shortcut -3
        self.convd4_4 = ConvBNAct(256, 256, 1, 1, bias=False, activation=activation, dw_sep_conv=False)
        # route -1 -10
        self.convd4_5 = ConvBNAct(512, 512, 1, 1, bias=False, activation=activation, dw_sep_conv=False)

        '''Downsample 5'''
        self.convd5_1 = ConvBNAct(512, 1024, kernel_size=3, stride=2, activation=activation,
                                  dw_sep_conv=dw_sep_conv)
        self.convd5_2 = ConvBNAct(1024, 512, kernel_size=1, stride=1, activation=activation,
                                  dw_sep_conv=False)
        # [route] layers=-2
        self.convd5_3 = ConvBNAct(1024, 512, kernel_size=1, stride=1, activation=activation,
                                  dw_sep_conv=dw_sep_conv)

        self.resblockd5 = DarkResBlock(512, n_blocks=4, activation=activation, dw_sep_conv=dw_sep_conv)

        # shortcut -3
        self.convd5_4 = ConvBNAct(512, 512, 1, 1, bias=False, activation=activation, dw_sep_conv=False)
        # route -1 -10
        self.convd5_5 = ConvBNAct(1024, 1024, 1, 1, bias=False, activation=activation, dw_sep_conv=False)

        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        xd1_1 = self.conv1(x)
        xd1_2 = self.convd1_1(xd1_1)
        xd1_3 = self.convd1_2(xd1_2)
        # route -2
        xd1_4 = self.convd1_3(xd1_2)
        xd1_5 = self.convd1_4(xd1_4)
        xd1_6 = self.convd1_5(xd1_5)
        # shortcut -3
        xd1_6 = xd1_6 + xd1_4
        xd1_7 = self.convd1_6(xd1_6)
        # route -1, -7
        xd1_7 = torch.cat([xd1_7, xd1_3], dim=1)
        down1 = self.convd1_7(xd1_7)

        xd2_1 = self.convd2_1(down1)
        xd2_2 = self.convd2_2(xd2_1)
        xd2_3 = self.convd2_3(xd2_1)
        res2 = self.resblockd2(xd2_3)
        xd2_4 = self.convd2_4(res2)
        xd2_4 = torch.cat([xd2_4, xd2_2], dim=1)
        down2 = self.convd2_5(xd2_4)

        xd3_1 = self.convd3_1(down2)
        xd3_2 = self.convd3_2(xd3_1)
        xd3_3 = self.convd3_3(xd3_1)
        res3 = self.resblockd3(xd3_3)
        xd3_4 = self.convd3_4(res3)
        xd3_4 = torch.cat([xd3_4, xd3_2], dim=1)
        down3 = self.convd3_5(xd3_4)

        xd4_1 = self.convd4_1(down3)
        xd4_2 = self.convd4_2(xd4_1)
        xd4_3 = self.convd4_3(xd4_1)
        res4 = self.resblockd4(xd4_3)
        xd4_4 = self.convd4_4(res4)
        xd4_4 = torch.cat([xd4_4, xd4_2], dim=1)
        down4 = self.convd4_5(xd4_4)

        xd5_1 = self.convd5_1(down4)
        xd5_2 = self.convd5_2(xd5_1)
        xd5_3 = self.convd5_3(xd5_1)
        res5 = self.resblockd5(xd5_3)
        xd5_4 = self.convd5_4(res5)
        xd5_4 = torch.cat([xd5_4, xd5_2], dim=1)
        down5 = self.convd5_5(xd5_4)
        pool = F.avg_pool2d(down5, 8)
        flat = torch.squeeze(pool)
        x_fc = self.fc(flat)
        out = F.log_softmax(x_fc, dim=0)
        return out


"""
    DENSENET
"""
densenet_121 = {
    "db1": 6,
    "db2": 12,
    "db3": 24,
    "db4": 16
}
densenet_169 = {
    "db1": 6,
    "db2": 12,
    "db3": 32,
    "db4": 32
}
densenet_201 = {
    "db1": 6,
    "db2": 12,
    "db3": 48,
    "db4": 32
}
densenet_264 = {
    "db1": 6,
    "db2": 12,
    "db3": 64,
    "db4": 48
}


class DenseNet(nn.Module):
    def __init__(self, dense_model, num_classes=1000, growth_rate=32, reduction=0.5, activation='relu',
                 dw_sep_conv=False):
        super(DenseNet, self).__init__()
        n_channels = 2 * growth_rate

        self.conv1 = nn.Conv2d(3, n_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.max_pool = nn.MaxPool2d(3, 2)
        self.dense1 = DenseBlock(n_channels, growth_rate, dense_model["db1"],
                                 activation=activation, dw_sep_conv=dw_sep_conv, bottleneck=True)
        n_channels += dense_model["db1"] * growth_rate
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans1 = TransitionLayer(n_channels, n_out_channels, activation=activation, dw_sep_conv=dw_sep_conv)

        n_channels = n_out_channels
        self.dense2 = DenseBlock(n_channels, growth_rate, dense_model["db2"],
                                 activation=activation, dw_sep_conv=dw_sep_conv, bottleneck=True)
        n_channels += dense_model["db2"] * growth_rate
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans2 = TransitionLayer(n_channels, n_out_channels, activation=activation, dw_sep_conv=dw_sep_conv)

        n_channels = n_out_channels
        self.dense3 = DenseBlock(n_channels, growth_rate, dense_model["db3"],
                                 activation=activation, dw_sep_conv=dw_sep_conv, bottleneck=True)
        n_channels += dense_model["db3"] * growth_rate
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans3 = TransitionLayer(n_channels, n_out_channels, activation=activation, dw_sep_conv=dw_sep_conv)

        n_channels = n_out_channels
        self.dense4 = DenseBlock(n_channels, growth_rate, dense_model["db4"],
                                 activation=activation, dw_sep_conv=dw_sep_conv, bottleneck=True)
        n_channels += dense_model["db4"] * growth_rate

        self.bn1 = nn.BatchNorm2d(n_channels)
        self.fc = nn.Linear(n_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(self.bn1(out), 9)
        out = torch.squeeze(out)
        out = self.fc(out)
        out = F.log_softmax(out, dim=0)
        return out


"""
    CSPDENSENET
"""

"""
    Test models
"""

if __name__ == '__main__':
    model_type = 'DenseNet'

    ngpu = 1

    image_size = 256
    batch_size = 4

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    if model_type == 'DarkNet53':
        net = DarkNet53(200, 'mish', dw_sep_conv=False).to(device)
    elif model_type == 'CSPDarkNet53':
        net = CSPDarkNet53(200, activation='mish', dw_sep_conv=False).to(device)
    elif model_type == 'DenseNet':
        net = DenseNet(densenet_121, num_classes=200, activation='mish', dw_sep_conv=False).to(device)
    else:
        print("Unsupported Model Type")
        exit(1)

    print(model_type + ": {}".format(net))

    randInput = torch.randn(batch_size, 3, image_size, image_size, device=device)

    output = net(randInput)
