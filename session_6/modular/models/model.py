#!/usr/bin/env python
"""model.py: This contains the model
definition used in session 6 to be trained on CIFAR10 dataset. """
from __future__ import print_function

import sys
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('./')

dropout_value = 0.029


class EVA8_session6_assignment_model(nn.Module):
    def __init__(self, normalization='batch'):
        super(EVA8_session6_assignment_model, self).__init__()
        self.normalization = normalization
        # C1 Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3),
                      padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )  # input:32x32x3, output:32x32x32, RF:3x3
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3),
                      stride=(2, 2), padding=1, bias=False, dilation=(2, 2)),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )  # input:32x32x32, output:15x15x64, RF:7x7

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1),
                      padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )  # input:15x15x64, output:17x17x32, RF:7x7

        # C2 Block
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3),
                      stride=(2, 2), padding=1, bias=False, dilation=(2, 2)),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )  # input:17x17x32, output:8x8x64, RF:15x15

        # TRANSITION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1),
                      padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )  # input:8x8x64, output:10x10x32, RF:15x15

        # C3 Block
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3),
                      stride=(2, 2), padding=1, bias=False, dilation=(2, 2)),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )  # input:10x10x32, output:4x4x64, RF:31x31

        # TRANSITION BLOCK 3
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1),
                      padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )  # input:4x4x64, output:6x6x32, RF:31x31

        # C4 Block
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3, 3),
                      padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )  # input:6x6x32, output:6x6x128, RF:47x47

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        )  # input:6x6x128, output:1x1x128, RF:87x87

        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=10, kernel_size=(1, 1),
                      padding=0, bias=False),
        )  # input:1x1x128, output:1x1x10,

    def forward(self, x):
        # C1 Block
        x = self.convblock1(x)
        x = self.convblock2(x)
        # TRANSITION BLOCK 1
        x = self.convblock3(x)
        # C2 Block
        x = self.convblock4(x)
        # TRANSITION BLOCK 2
        x = self.convblock5(x)
        # C3 Block
        x = self.convblock6(x)
        # TRANSITION BLOCK 3
        x = self.convblock7(x)
        # C4 Block
        x = self.convblock8(x)
        # OUTPUT BLOCK
        x = self.gap(x)
        x = self.convblock9(x)
        # Reshape
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)  # torch.nn.CrossEntropyLoss:criterion
        # combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
