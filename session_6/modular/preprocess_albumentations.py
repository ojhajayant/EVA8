#!/usr/bin/env python
"""
preprocess_albumentations.py: This contains the data-pre-processing routine
implemented using albumentations library.
https://github.com/albumentations-team/albumentations
"""
from __future__ import print_function

import sys
import cv2

import numpy as np
import torch
from albumentations import Compose, RandomCrop, HorizontalFlip, Normalize
from albumentations import CoarseDropout, PadIfNeeded, ShiftScaleRotate
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import datasets

from session_6.modular import cfg

sys.path.append('./')
global args
args = cfg.args

file_path = args.data


# IPYNB_ENV = True  # By default ipynb notebook env

class album_Compose:
    def __init__(self,
                 img_size,
                 train=True,
                 mean=[0.49139968, 0.48215841, 0.44653091],
                 std=[0.24703223, 0.24348513, 0.26158784]
                 ):
        if train:
            self.albumentations_transform = Compose([
                PadIfNeeded(min_height=img_size[0], min_width=img_size[1],
                            border_mode=cv2.BORDER_REPLICATE,
                            always_apply=True),
                CoarseDropout(max_holes=1, max_height=img_size[0] // 2,
                              max_width=img_size[1] // 2,
                              min_height=img_size[0] // 2,
                              min_width=img_size[1] // 2,
                              always_apply=False, p=0.5,
                              fill_value=tuple([x * 255.0 for x in mean])),
                HorizontalFlip(p=0.5),
                ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                                 rotate_limit=10,
                                 border_mode=cv2.BORDER_CONSTANT, value=0),
                RandomCrop(height=32, width=32, always_apply=True),
                Normalize(mean=mean, std=std, always_apply=True),
                ToTensorV2(),

            ])
        else:
            self.albumentations_transform = Compose([
                Normalize(mean=mean, std=std, always_apply=True),
                ToTensorV2(),

            ])

    def __call__(self, img):
        img = np.array(img)
        img = self.albumentations_transform(image=img)['image']
        return img


def preprocess_data_albumentations(mean_tuple, std_tuple, img_size):
    """
    Used for pre-processing the data
    when for args.use_albumentations True
    """
    # Train Phase transformations
    global args
    # tensor_args1 = dict(always_apply=True, p=1.0)
    tensor_args2 = dict(num_classes=1, sigmoid=True, normalize=None)
    norm_args = dict(mean=mean_tuple, std=std_tuple, max_pixel_value=255.0,
                     always_apply=False, p=1.0)
    print("************")
    train_transforms = album_Compose(img_size, train=True, mean=mean_tuple,
                                     std=std_tuple)

    # Test Phase transformations
    test_transforms = album_Compose(img_size, train=False, mean=mean_tuple,
                                    std=std_tuple)

    train_kwargs = dict(train=True, download=True, transform=train_transforms)
    test_kwargs = dict(train=False, download=True, transform=test_transforms)
    if args.dataset == 'CIFAR10':
        train_dataset = datasets.CIFAR10(file_path, **train_kwargs)
        test_dataset = datasets.CIFAR10(file_path, **test_kwargs)
    elif args.dataset == 'MNIST':
        train_dataset = datasets.MNIST(file_path, **train_kwargs)
        test_dataset = datasets.MNIST(file_path, **test_kwargs)

    print("CUDA Available?", args.cuda)

    # For reproducibility
    torch.manual_seed(args.SEED)

    if args.cuda:
        torch.cuda.manual_seed(args.SEED)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=args.batch_size,
                           num_workers=4,
                           pin_memory=True) if args.cuda else \
        dict(shuffle=True, batch_size=args.batch_size)
    # train dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_args)
    # test dataloader
    test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_args)
    return train_dataset, test_dataset, train_loader, test_loader
