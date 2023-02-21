
# Session_8

### 

>   Write a custom ResNet architecture for CIFAR10 that has the following architecture:

*  PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]


*  Layer1 -

    >    X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]

    >    R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 

    >    Add(X, R1)


*  Layer 2 -

    > Conv 3x3 [256k]

    > MaxPooling2D

    > BN

    > ReLU


* Layer 3 -

    > X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]

    > R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]

    > Add(X, R2)


*  MaxPooling with Kernel Size 4


* FC Layer 


* SoftMax




*   Uses One Cycle Policy such that:

  > Total Epochs = 24

  > Max at Epoch = 5

  > LRMIN = FIND

  > LRMAX = FIND

  > NO Annihilation

* Uses this transform 
  > -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)

  > Batch size = 512

  > Target Accuracy: 90% (93.8% quadruple scores). 


1.   [EVA8_API](https://github.com/ojhajayant/EVA8_API) is the main repo which is being cloned here to be able to run the main.py script with various user provided (or default) arg options.

2.   The LR-Range test over 500 epochs, as required for the One Cycle Policy (OCP) gave a max_lr(best_lr) of 0.022000000097799996, I am using a LR which is a little ahead in terms of rounding off (and also due to the accuracy Vs learning rate plot obtained from the LR-range test run below in this notebook)

3.   Trained for EPOCHS= 24 epochs as required, the div_factor was taken as 10, so as to start the cycle with a learning rate of best_lr/10 = 0.003, it is required that the max LR is reached on 5th epoch, with NO annihilation epochs, hence final_div_factor = div_factor & MAX_LR_EPOCH = 5 thus resulting in PCT_START = MAX_LR_EPOCH / EPOCHS = 0.2

4. Here are the different args values for this run:

	cmd : train
	IPYNB_ENV : True
	use_albumentations : True
	SEED : 1
	dataset : CIFAR10
	img_size : (32, 32)
	batch_size : 512
	epochs : 24
	criterion : CrossEntropyLoss()
	init_lr : 0.0001
	end_lr : 0.1
	max_lr_epochs : 5
	lr_range_test_epochs : 10
	best_lr : 0.040031316333168206
	cycle_momentum : True
	div_factor : 6500
	optimizer : <class 'torch.optim.sgd.SGD'>
	cuda : True
	dropout : 0.08
	l1_weight : 2.5e-05
	l2_weight_decay : 0.0002125
	L1 : True
	L2 : False
	data : ./data/
	best_model_path : ./saved_models/
	prefix : data
	best_model :  CIFAR10_model_epoch-24_L1-1_L2-0_val_acc-89.23.h5


5.  max test/validation accuracy within 24 epochs = ~89.23%








Please refer the [notebook](https://github.com/ojhajayant/EVA8/blob/main/session_8/EVA8_session8_assignment.ipynb) for this assignment solution.

- max test/validation accuracy within 24 epochs = 89.17%

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
            Conv2d-4          [-1, 128, 32, 32]          73,728
         MaxPool2d-5          [-1, 128, 16, 16]               0
       BatchNorm2d-6          [-1, 128, 16, 16]             256
              ReLU-7          [-1, 128, 16, 16]               0
      HighwayBlock-8          [-1, 128, 16, 16]               0
            Conv2d-9          [-1, 128, 16, 16]         147,456
      BatchNorm2d-10          [-1, 128, 16, 16]             256
             ReLU-11          [-1, 128, 16, 16]               0
           Conv2d-12          [-1, 128, 16, 16]         147,456
      BatchNorm2d-13          [-1, 128, 16, 16]             256
             ReLU-14          [-1, 128, 16, 16]               0
         ResBlock-15          [-1, 128, 16, 16]               0
            Layer-16          [-1, 128, 16, 16]               0
           Conv2d-17          [-1, 256, 16, 16]         294,912
        MaxPool2d-18            [-1, 256, 8, 8]               0
      BatchNorm2d-19            [-1, 256, 8, 8]             512
             ReLU-20            [-1, 256, 8, 8]               0
     HighwayBlock-21            [-1, 256, 8, 8]               0
            Layer-22            [-1, 256, 8, 8]               0
           Conv2d-23            [-1, 512, 8, 8]       1,179,648
        MaxPool2d-24            [-1, 512, 4, 4]               0
      BatchNorm2d-25            [-1, 512, 4, 4]           1,024
             ReLU-26            [-1, 512, 4, 4]               0
     HighwayBlock-27            [-1, 512, 4, 4]               0
           Conv2d-28            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-29            [-1, 512, 4, 4]           1,024
             ReLU-30            [-1, 512, 4, 4]               0
           Conv2d-31            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-32            [-1, 512, 4, 4]           1,024
             ReLU-33            [-1, 512, 4, 4]               0
         ResBlock-34            [-1, 512, 4, 4]               0
            Layer-35            [-1, 512, 4, 4]               0
        MaxPool2d-36            [-1, 512, 1, 1]               0
           Linear-37                   [-1, 10]           5,120
================================================================
Total params: 6,573,120
Trainable params: 6,573,120
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 7.63
Params size (MB): 25.07
Estimated Total Size (MB): 32.72
----------------------------------------------------------------
```

LR Range Test Logs 
Please note for the LR Range test plot points to the location of the best_lr 
(i.e. the point where the accuracy value reached max, i,e, MAX-LR value for OCP)
as 0.022000000097799996, given the plot and some experiments with both 
0.022000000097799996 & its nearest rounded off value 0.03, I am choosing this 
value of 0.03 as, with a resulting start of 0.003 for a 5 epoch run, it will 
approximately reach, the found out max-lr/best_lr value of approx: 0.022 as 
with PCT_START of 0.2 it will require ~4.8 epochs to reach highest which happens 
to be around same: i.e. in a "train" run, at EPOCH: 5 --> we would reach 
LR: 0.024644171779141102 (as appears under the training logs here)

```

Training Logs:

```
Model training starts on CIFAR10 dataset
EPOCH: 1
LR: 6.158664051256647e-06
Loss=3.5662190914154053 Batch_id=97 Accuracy=25.71: 100%|██████████| 98/98 [00:04<00:00, 22.13it/s]
Test set: Average loss: 0.0041, Accuracy: 3935/10000 (39.35%)

validation-accuracy improved from 0 to 39.35, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-1_L1-1_L2-0_val_acc-39.35.h5
EPOCH: 2
LR: 0.008027560405510276
Loss=3.3360891342163086 Batch_id=97 Accuracy=45.80: 100%|██████████| 98/98 [00:04<00:00, 23.77it/s]
Test set: Average loss: 0.0039, Accuracy: 4905/10000 (49.05%)

validation-accuracy improved from 39.35 to 49.05, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-2_L1-1_L2-0_val_acc-49.05.h5
EPOCH: 3
LR: 0.016048962146969298
Loss=3.1999800205230713 Batch_id=97 Accuracy=54.49: 100%|██████████| 98/98 [00:03<00:00, 25.54it/s]
Test set: Average loss: 0.0039, Accuracy: 5259/10000 (52.59%)

validation-accuracy improved from 49.05 to 52.59, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-3_L1-1_L2-0_val_acc-52.59.h5
EPOCH: 4
LR: 0.024070363888428318
Loss=3.08453369140625 Batch_id=97 Accuracy=60.07: 100%|██████████| 98/98 [00:03<00:00, 25.51it/s]
Test set: Average loss: 0.0037, Accuracy: 5940/10000 (59.40%)

validation-accuracy improved from 52.59 to 59.4, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-4_L1-1_L2-0_val_acc-59.4.h5
EPOCH: 5
LR: 0.03209176562988734
Loss=2.9640989303588867 Batch_id=97 Accuracy=65.32: 100%|██████████| 98/98 [00:03<00:00, 25.91it/s]
Test set: Average loss: 0.0036, Accuracy: 6465/10000 (64.65%)

validation-accuracy improved from 59.4 to 64.65, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-5_L1-1_L2-0_val_acc-64.65.h5
EPOCH: 6
LR: 0.040009817237902
Loss=2.8886799812316895 Batch_id=97 Accuracy=68.02: 100%|██████████| 98/98 [00:03<00:00, 26.14it/s]
Test set: Average loss: 0.0035, Accuracy: 7061/10000 (70.61%)

validation-accuracy improved from 64.65 to 70.61, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-6_L1-1_L2-0_val_acc-70.61.h5
EPOCH: 7
LR: 0.0379029059018135
Loss=2.7771425247192383 Batch_id=97 Accuracy=71.00: 100%|██████████| 98/98 [00:04<00:00, 24.42it/s]
Test set: Average loss: 0.0035, Accuracy: 6956/10000 (69.56%)

EPOCH: 8
LR: 0.035795994565725006
Loss=2.682971954345703 Batch_id=97 Accuracy=72.86: 100%|██████████| 98/98 [00:03<00:00, 25.25it/s]
Test set: Average loss: 0.0034, Accuracy: 7413/10000 (74.13%)

validation-accuracy improved from 70.61 to 74.13, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-8_L1-1_L2-0_val_acc-74.13.h5
EPOCH: 9
LR: 0.033689083229636506
Loss=2.5929462909698486 Batch_id=97 Accuracy=74.33: 100%|██████████| 98/98 [00:03<00:00, 24.65it/s]
Test set: Average loss: 0.0035, Accuracy: 7117/10000 (71.17%)

EPOCH: 10
LR: 0.03158217189354801
Loss=2.4941048622131348 Batch_id=97 Accuracy=77.90: 100%|██████████| 98/98 [00:03<00:00, 25.85it/s]
Test set: Average loss: 0.0034, Accuracy: 7813/10000 (78.13%)

validation-accuracy improved from 74.13 to 78.13, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-10_L1-1_L2-0_val_acc-78.13.h5
EPOCH: 11
LR: 0.029475260557459512
Loss=2.422914981842041 Batch_id=97 Accuracy=79.79: 100%|██████████| 98/98 [00:03<00:00, 25.70it/s]
Test set: Average loss: 0.0034, Accuracy: 7852/10000 (78.52%)

validation-accuracy improved from 78.13 to 78.52, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-11_L1-1_L2-0_val_acc-78.52.h5
EPOCH: 12
LR: 0.02736834922137102
Loss=2.375701427459717 Batch_id=97 Accuracy=80.85: 100%|██████████| 98/98 [00:03<00:00, 24.92it/s]
Test set: Average loss: 0.0033, Accuracy: 8084/10000 (80.84%)

validation-accuracy improved from 78.52 to 80.84, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-12_L1-1_L2-0_val_acc-80.84.h5
EPOCH: 13
LR: 0.025261437885282522
Loss=2.30702805519104 Batch_id=97 Accuracy=82.09: 100%|██████████| 98/98 [00:03<00:00, 25.78it/s]
Test set: Average loss: 0.0033, Accuracy: 8174/10000 (81.74%)

validation-accuracy improved from 80.84 to 81.74, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-13_L1-1_L2-0_val_acc-81.74.h5
EPOCH: 14
LR: 0.023154526549194025
Loss=2.2900397777557373 Batch_id=97 Accuracy=83.35: 100%|██████████| 98/98 [00:03<00:00, 25.61it/s]
Test set: Average loss: 0.0033, Accuracy: 8279/10000 (82.79%)

validation-accuracy improved from 81.74 to 82.79, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-14_L1-1_L2-0_val_acc-82.79.h5
EPOCH: 15
LR: 0.02104761521310553
Loss=2.224611282348633 Batch_id=97 Accuracy=83.74: 100%|██████████| 98/98 [00:03<00:00, 25.39it/s]
Test set: Average loss: 0.0033, Accuracy: 8286/10000 (82.86%)

validation-accuracy improved from 82.79 to 82.86, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-15_L1-1_L2-0_val_acc-82.86.h5
EPOCH: 16
LR: 0.018940703877017035
Loss=2.209446668624878 Batch_id=97 Accuracy=84.62: 100%|██████████| 98/98 [00:03<00:00, 25.77it/s]
Test set: Average loss: 0.0033, Accuracy: 8280/10000 (82.80%)

EPOCH: 17
LR: 0.01683379254092854
Loss=2.1634647846221924 Batch_id=97 Accuracy=85.18: 100%|██████████| 98/98 [00:03<00:00, 25.10it/s]
Test set: Average loss: 0.0033, Accuracy: 8274/10000 (82.74%)

EPOCH: 18
LR: 0.014726881204840041
Loss=2.121339797973633 Batch_id=97 Accuracy=86.09: 100%|██████████| 98/98 [00:03<00:00, 26.03it/s]
Test set: Average loss: 0.0032, Accuracy: 8437/10000 (84.37%)

validation-accuracy improved from 82.86 to 84.37, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-18_L1-1_L2-0_val_acc-84.37.h5
EPOCH: 19
LR: 0.012619969868751541
Loss=2.116539716720581 Batch_id=97 Accuracy=86.63: 100%|██████████| 98/98 [00:03<00:00, 25.27it/s]
Test set: Average loss: 0.0032, Accuracy: 8371/10000 (83.71%)

EPOCH: 20
LR: 0.010513058532663048
Loss=2.081789493560791 Batch_id=97 Accuracy=87.52: 100%|██████████| 98/98 [00:03<00:00, 25.93it/s]
Test set: Average loss: 0.0032, Accuracy: 8592/10000 (85.92%)

validation-accuracy improved from 84.37 to 85.92, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-20_L1-1_L2-0_val_acc-85.92.h5
EPOCH: 21
LR: 0.008406147196574551
Loss=2.0503764152526855 Batch_id=97 Accuracy=88.05: 100%|██████████| 98/98 [00:03<00:00, 25.97it/s]
Test set: Average loss: 0.0032, Accuracy: 8685/10000 (86.85%)

validation-accuracy improved from 85.92 to 86.85, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-21_L1-1_L2-0_val_acc-86.85.h5
EPOCH: 22
LR: 0.006299235860486058
Loss=2.061082124710083 Batch_id=97 Accuracy=89.16: 100%|██████████| 98/98 [00:04<00:00, 24.14it/s]
Test set: Average loss: 0.0032, Accuracy: 8710/10000 (87.10%)

validation-accuracy improved from 86.85 to 87.1, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-22_L1-1_L2-0_val_acc-87.1.h5
EPOCH: 23
LR: 0.004192324524397557
Loss=2.0236897468566895 Batch_id=97 Accuracy=90.25: 100%|██████████| 98/98 [00:03<00:00, 25.86it/s]
Test set: Average loss: 0.0032, Accuracy: 8824/10000 (88.24%)

validation-accuracy improved from 87.1 to 88.24, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-23_L1-1_L2-0_val_acc-88.24.h5
EPOCH: 24
LR: 0.002085413188309064
Loss=1.9778934717178345 Batch_id=97 Accuracy=91.24: 100%|██████████| 98/98 [00:03<00:00, 25.60it/s]
Test set: Average loss: 0.0031, Accuracy: 8923/10000 (89.23%)

validation-accuracy improved from 88.24 to 89.23, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-24_L1-1_L2-0_val_acc-89.23.h5
```


#### Few details on the Model:

Please refer [custom_resnet.py](https://github.com/ojhajayant/EVA8_API/blob/main/models/custom_resnet.py)
All the model files now reside in the same location like this custom_resnet.py file

```python
!git clone https://git@github.com/ojhajayant//EVA8_API.git
 ```


###  Following graph shows the model accuracy:

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_8/model_acc.png "Logo Title Text 1")

###  Following graph shows the change in momentum & LR for the Once Cycle Policy across different iterations:

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_8/momentum_lr.png "Logo Title Text 1")


###  Misclassified images:


For Best saved Model :

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_8/misclassified%20images.png "Logo Title Text 1")

With Grad Cam heatmap :

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_8/grad%20cam%20misclassified%20images.png "Logo Title Text 1")



###  Confusion Matrix & Classification Reports:

For Best saved Model:

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_8/confusion_matrix%20classification%20rpt.png "Logo Title Text 1")















