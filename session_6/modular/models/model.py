#!/usr/bin/env python
"""model.py: This contains the model
definition used in session 6 to be trained on CIFAR10 dataset. """
from __future__ import print_function

import sys
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('./')


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation=dilation,groups=in_channels,bias=bias),
            nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias),
        )
    def forward(self,x):
        x = self.convblock(x)
        return x

dropout_value = 0.05
class EVA8_session6_assignment_model(nn.Module):
    def __init__(self, normalization='batch'):
        super(EVA8_session6_assignment_model, self).__init__()
        self.normalization = normalization
        # C1 Block
        self.convblock1 = nn.Sequential(
            SeparableConv2d(in_channels=3, out_channels=64, kernel_size=(3, 3),
                      padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )  # input:32x32x3, output:32x32x64, RF:3x3
        self.convblock2 = nn.Sequential(
            SeparableConv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                      stride=2, padding=2, bias=False, dilation=2),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )  # input:32x32x64, output:16x16x128, RF:7x7

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            SeparableConv2d(in_channels=128, out_channels=64, kernel_size=(1, 1),
                      padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )  # input:16x16x128, output:16x16x64, RF:7x7

        # C2 Block
        self.convblock4 = nn.Sequential(
            SeparableConv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                      padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )  # input:16x16x64, output:16x16x128, RF:11x11
        self.convblock5 = nn.Sequential(
            SeparableConv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                      stride=(2, 2), padding=2, bias=False, dilation=(2, 2)),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )  # input:16x16x128, output:8x8x128, RF:19x19

        # TRANSITION BLOCK 2
        self.convblock6 = nn.Sequential(
            SeparableConv2d(in_channels=128, out_channels=64, kernel_size=(1, 1),
                      padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )  # input:8x8x128, output:8x8x64, RF:19x19

        # C3 Block
        self.convblock7 = nn.Sequential(
            SeparableConv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                      padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )  # input:8x8x64, output:8x8x128, RF:27x27
        self.convblock8 = nn.Sequential(
            SeparableConv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                      padding=1, bias=False, dilation=(2, 2)),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )  # input:8x8x128, output:6x6x128, RF:43x43

        # TRANSITION BLOCK 3
        self.convblock9 = nn.Sequential(
            SeparableConv2d(in_channels=128, out_channels=64, kernel_size=(1, 1),
                      padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )  # input:6x6x128, output:6x6x64, RF:43x43

        # C4 Block
        self.convblock10 = nn.Sequential(
            SeparableConv2d(in_channels=64, out_channels=256, kernel_size=(3, 3),
                      padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )  # input:6x6x64, output:6x6x256, RF:51x51

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        )  # input:6x6x256, output:1x1x256, RF:71x71

        self.convblock11 = nn.Sequential(
            SeparableConv2d(in_channels=256, out_channels=10, kernel_size=(1, 1),
                      padding=0, bias=False),
        )  # input:1x1x256, output:1x1x10,

    def forward(self, x):
        # C1 Block
        x = self.convblock1(x)
        x = self.convblock2(x)
        # TRANSITION BLOCK 1
        x = self.convblock3(x)
        # C2 Block
        x = self.convblock4(x)
        x = self.convblock5(x)
        # TRANSITION BLOCK 2
        x = self.convblock6(x)
        # C3 Block
        x = self.convblock7(x)
        x = self.convblock8(x)
        # TRANSITION BLOCK 3
        x = self.convblock9(x)
        # C4 Block
        x = self.convblock10(x)
        # OUTPUT BLOCK
        x = self.gap(x)
        x = self.convblock11(x)
        # Reshape
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)  # torch.nn.CrossEntropyLoss:criterion
        # combines nn.LogSoftmax() and nn.NLLLoss() in one single class.) #torch.nn.CrossEntropyLoss:criterion combines
                                        #nn.LogSoftmax() and nn.NLLLoss() in one single class.

