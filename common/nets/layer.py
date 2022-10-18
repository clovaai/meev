import math

import torch
import torch.nn as nn
from torch.nn import functional as F


def make_linear_layers(feat_dims, relu_final=True, use_bn=False):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i+1]))

        # Do not use ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and relu_final):
            if use_bn:
                layers.append(nn.BatchNorm1d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_conv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


def make_dwconv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True) :
    layers = []
    for i in range(len(feat_dims)-1) :
        layers.append(
            nn.Conv2d(
                kernel_size=kernel,
                in_channels=feat_dims[i],
                out_channels=feat_dims[i],
                stride=stride,
                padding=padding,
                groups=feat_dims[i]))
        layers.append(nn.BatchNorm2d(feat_dims[i]))
        layers.append(nn.ReLU(inplace=True))
        #pointwise
        layers.append(nn.Conv2d(feat_dims[i], feat_dims[i+1], kernel_size=1, stride=1, padding=0))

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',  nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, act_method = "relu6", norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        if act_method == "relu6":
            act_layer = nn.ReLU6(inplace=True)
        elif act_method == "relu":
            act_layer = nn.ReLU(inplace=True)
        else:
            act_layer = nn.ReLU(inplace=True)

        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            act_layer
        )
        

class DWConvLayer(nn.Module):
    def __init__(self, n_in, n_out, stride=1, pooling = None, last_act=True, act_method = "relu6", kernel_size=3, padding=None, batch_norm=True):
        super(DWConvLayer, self).__init__()
        layers = []
        #depthwise
        if padding is None:
            padding = 1
            if kernel_size == 5:
                padding = 2
            if kernel_size == 7:
                padding = 3
        layers.append(
            nn.Conv2d(n_in, n_in, kernel_size=kernel_size, padding=padding, stride=stride, groups=n_in))
        if batch_norm:
            layers.append(nn.BatchNorm2d(n_in))
        if act_method == "relu6":
            layers.append(nn.ReLU6(inplace=True))
        elif act_method == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif act_method == "lrelu":
            layers.append(nn.LeakyReLU(inplace=True))
        else:
            layers.append(nn.ReLU(inplace=True))

        #pointwise
        layers.append(nn.Conv2d(n_in, n_out, kernel_size=1, stride=1, padding=0))

        if pooling is not None:
            if pooling == "max":
                layers.append(nn.MaxPool2d(kernel_size=2))
            elif pooling == "avg":
                layers.append(torch.nn.AvgPool2d(kernel_size=2))

        if last_act:
            layers.append(nn.BatchNorm2d(n_out))
            if act_method == "relu6":
                layers.append(nn.ReLU6(inplace=True))
            elif act_method == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif act_method == "lrelu":
                layers.append(nn.LeakyReLU(inplace=True))
            else:
                layers.append(nn.ReLU(inplace=True))
        
        self.conv = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',  nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.conv(x)

