#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import warnings

import numpy as np
import torch
import torch.optim as optim
from torchsummary import summary

# sys.path.append('./')

from session_5.modular.models import EVA8_session4_assignment_model
from session_5.modular import cfg
from session_5.modular import preprocess
from session_5.modular import test
from session_5.modular import train
from session_5.modular import utils

global args
args = cfg.args
if args.cmd == None:
    args.cmd = 'train'



def main_EVA8_session4_assignment_model():
    global args
    print("The config used for this run are being saved @ {}".format(os.path.
        join(
        args.prefix, 'config_params.txt')))
    utils.write(vars(args), os.path.join(args.prefix, 'config_params.txt'))

    mean, std = preprocess.get_dataset_mean_std()
    if not isinstance(mean, tuple):
        train_dataset, test_dataset, train_loader, test_loader = preprocess.preprocess_data(
            (mean,), (std,))

    preprocess.get_data_stats(train_dataset, test_dataset, train_loader)
    utils.plot_train_samples(train_loader)
    L1 = args.L1
    L2 = args.L2
    device = torch.device("cuda" if args.cuda else "cpu")
    print(device)
    model = EVA8_session4_assignment_model.Net()
    model = model.to(device)
    # model = EVA8_session4_assignment_model().to(device)
    if args.dataset == 'MNIST':
        summary(model, input_size=(1, 28, 28))
    if args.cmd == 'train':
        print("Model training starts on {} dataset".format(args.dataset))
        # Enable L2-regularization with supplied value of weight decay, or keep it default-0
        if L2:
            weight_decay = args.l2_weight_decay
        else:
            weight_decay = 0
        # lr = args.lr
        lr = 0.01
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                              weight_decay=weight_decay)

        EPOCHS = args.epochs
        for epoch in range(EPOCHS):
            print("EPOCH:", epoch + 1)
            train.train(model, device, train_loader, optimizer, epoch)
            test.test(model, device, test_loader, optimizer, epoch)
        utils.plot_acc_loss()
    elif args.cmd == 'test':
        print("Model inference starts on {}  dataset".format(args.dataset))
        # model_name = args.best_model
        model_name = 'MNIST_model_epoch-8_L1-1_L2-0_val_acc-99.26.h5'
        print("Loaded the best model: {} from last training session".format(
            model_name))
        model = utils.load_model(s5_s6_custom_model_mnist.Net(), device,
                                 model_name=model_name)
        y_test = np.array(test_dataset.targets)
        print(
            "The confusion-matrix and classification-report for this model are:")
        y_pred = utils.model_pred(model, device, y_test, test_dataset)
        x_test = test_dataset.data
        utils.display_mislabelled(model, device, x_test, y_test.reshape(-1, 1),
                                  y_pred, test_dataset,
                                  title_str='Predicted Vs Actual With L1')


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    args.dataset = 'MNIST'
    args.cmd = 'train'
    args.IPYNB_ENV = 'False'
    args.epochs = 1
    main_EVA8_session4_assignment_model()

