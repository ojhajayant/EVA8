
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

2.   The LR-Range test, as required for the One Cycle Policy (OCP) gave a max_lr(best_lr) of 4.59E-02

3.   Trained for EPOCHS= 24 epochs as required, the div_factor was taken as 10,  it is required that the max LR is reached on 5th epoch, with NO annihilation epochs, hence final_div_factor = div_factor & MAX_LR_EPOCH = 5 thus resulting in PCT_START = MAX_LR_EPOCH / EPOCHS = 0.2

4.  Here are the different args values for this run:

	> cmd : test

	> IPYNB_ENV : True

	> use_albumentations : True

	> SEED : 1

	> dataset : CIFAR10

	> img_size : (32, 32)

	> batch_size : 512

	> epochs : 24

	> criterion : CrossEntropyLoss()

	> init_lr : 0.0001

	> end_lr : 1

	> max_lr_epochs : 5

	> lr_range_test_epochs : 10

	> best_lr : 0.03

	> cycle_momentum : True

	> div_factor : 10

	> optimizer : <class 'torch.optim.sgd.SGD'>

	> cuda : True

	> dropout : 0.08

	> l1_weight : 2.5e-05

	> l2_weight_decay : 0.0002125

	> L1 : True

	> L2 : False

	> data : ./data/

	> best_model_path : ./saved_models/

	>  prefix : data

	> best_model : CIFAR10_model_epoch-24_L1-1_L2-0_val_acc-91.78.h5


5.  max test/validation accuracy within 24 epochs = 91.78%














Please refer the [notebook](https://github.com/ojhajayant/EVA8/blob/main/session_8/EVA8_session8_assignment.ipynb) for this assignment solution.

- max test/validation accuracy within 24 epochs = 91.78%

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

# Summary of Steps

1.   Range test run over 10 epochs, between 1E-05 to 4 (will provide 2 values: 
at a "steepest gradient" point & 2nd at the lowest loss value.

	  	> i.e.   %run /content/EVA8_API/main.py --cmd lr_find --init_lr 1e-5 --end_lr 4 --lr_range_test_epochs 10

	  	>  Got 2 LR points : 1st "steepest gradient" Suggested LR: 4.59E-02 & 2nd "min loss": 0.6492212094700427
		
		
2.   Run the "train" command using above 2 values from the above run:  first a "train" run on the steepest gradient 
     point-> i.e. 4.59E-02

	  	>  %run  /content/EVA8_API/main.py --cmd train --best_lr 4.59E-02 --L1=True --cycle_momentum=True --div_factor=10
		
	  	> Thus, the "steepest gradient" point LR value found by sweeping the range 1E-05 to 4 over 10 epochs was: 
	  	  4.59E-02, which gives 91.44% max, now below we run "train" command using the  "min-loss" point LR value (2nd 
	  	  point mentioned earlier)
		  
		  
3.    Run the "train" command with the min-loss point-> i.e. 0.6492212094700427

	  	> i.e. %run  /content/EVA8_API/main.py --cmd train --best_lr 0.6492212094700427 --L1=True --cycle_momentum=True
	  	  --div_factor=10
		
	  	> "min-loss" point LR value  0.6492212094700427,  gives a very low: 71.88% max,
		

4.    Hence now we zoom-in within the range (4.59E-02)/10 to 10*(4.59E-02) over 40 epochs

	  	> i.e. %run /content/EVA8_API/main.py --cmd lr_find --init_lr 4.59E-03 --end_lr 4.59E-01 --lr_range_test_epochs 40
		
	  	>  Got 2 LR points : 1st "steepest gradient" Suggested LR: 4.68E-03 & 2nd "min loss": 0.459
		
		
		
5.   Now below we run "train" command using this new steepest gradient point first: 4.68E-03

	  	> i.e. %run  /content/EVA8_API/main.py --cmd train --best_lr 4.68E-03 --L1=True --cycle_momentum=True --div_factor=10
		
	  	> This new "stepest gradient" point gives just 87.88% as max
		

6.   Hence now we will be fixing the best_lr/max_lr as 4.59E-02 (which had already given a max of 91.44%)

7.   As one more step, now we fix our max_LR/Best_lr as 4.59E-02 and just as an additional transform we 
     add the ShiftScaleRotate(shift_limit=0.07, scale_limit=0.2, rotate_limit=15, border_mode=cv2.BORDER_WRAP)
     apart from the other 3 transforms used for the earlier run i.e. RandomCrop 32, 32 (after padding of 4) -- FlipLR --Followed by 
     CutOut(8, 8)....(PLEASE NOTE: FOR THE LR_FINDER RANGE TEST OPERATION THESE TRANSFORMS HAVE NOT BEEN USED)
     
		  > i.e. %run  /content/EVA8_API/main.py --cmd train --best_lr 4.59E-02 --L1=True --cycle_momentum=True --div_factor=10
		
		  > Now the max accuracy improves a little to 91.78%


Training Logs:

```
Model training starts on CIFAR10 dataset
EPOCH: 1
LR: 0.00459
Loss=2.9385342597961426 Batch_id=97 Accuracy=33.85: 100%|██████████| 98/98 [00:04<00:00, 20.05it/s]

Test set: Average loss: 0.0028, Accuracy: 4908/10000 (49.08%)

validation-accuracy improved from 0 to 49.08, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-1_L1-1_L2-0_val_acc-49.08.h5
EPOCH: 2
LR: 0.012868895705521473
Loss=2.513735771179199 Batch_id=97 Accuracy=51.38: 100%|██████████| 98/98 [00:04<00:00, 22.66it/s]

Test set: Average loss: 0.0028, Accuracy: 5364/10000 (53.64%)

validation-accuracy improved from 49.08 to 53.64, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-2_L1-1_L2-0_val_acc-53.64.h5
EPOCH: 3
LR: 0.021147791411042945
Loss=2.4084105491638184 Batch_id=97 Accuracy=60.84: 100%|██████████| 98/98 [00:04<00:00, 23.37it/s]

Test set: Average loss: 0.0031, Accuracy: 5838/10000 (58.38%)

validation-accuracy improved from 53.64 to 58.38, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-3_L1-1_L2-0_val_acc-58.38.h5
EPOCH: 4
LR: 0.029426687116564414
Loss=2.2699570655822754 Batch_id=97 Accuracy=64.73: 100%|██████████| 98/98 [00:04<00:00, 22.90it/s]

Test set: Average loss: 0.0029, Accuracy: 6151/10000 (61.51%)

validation-accuracy improved from 58.38 to 61.51, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-4_L1-1_L2-0_val_acc-61.51.h5
EPOCH: 5
LR: 0.03770558282208589
Loss=2.0248522758483887 Batch_id=97 Accuracy=68.37: 100%|██████████| 98/98 [00:04<00:00, 22.52it/s]

Test set: Average loss: 0.0020, Accuracy: 6918/10000 (69.18%)

validation-accuracy improved from 61.51 to 69.18, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-5_L1-1_L2-0_val_acc-69.18.h5
EPOCH: 6
LR: 0.0458755955961332
Loss=1.7590603828430176 Batch_id=97 Accuracy=73.47: 100%|██████████| 98/98 [00:04<00:00, 22.92it/s]

Test set: Average loss: 0.0012, Accuracy: 7935/10000 (79.35%)

validation-accuracy improved from 69.18 to 79.35, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-6_L1-1_L2-0_val_acc-79.35.h5
EPOCH: 7
LR: 0.04348396401718582
Loss=1.6039888858795166 Batch_id=97 Accuracy=76.64: 100%|██████████| 98/98 [00:04<00:00, 22.52it/s]

Test set: Average loss: 0.0014, Accuracy: 7865/10000 (78.65%)

EPOCH: 8
LR: 0.041092332438238455
Loss=1.6321141719818115 Batch_id=97 Accuracy=78.76: 100%|██████████| 98/98 [00:04<00:00, 22.69it/s]

Test set: Average loss: 0.0010, Accuracy: 8271/10000 (82.71%)

validation-accuracy improved from 79.35 to 82.71, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-8_L1-1_L2-0_val_acc-82.71.h5
EPOCH: 9
LR: 0.03870070085929109
Loss=1.494049310684204 Batch_id=97 Accuracy=79.73: 100%|██████████| 98/98 [00:04<00:00, 22.79it/s]

Test set: Average loss: 0.0011, Accuracy: 8231/10000 (82.31%)

EPOCH: 10
LR: 0.03630906928034372
Loss=1.4807716608047485 Batch_id=97 Accuracy=80.87: 100%|██████████| 98/98 [00:04<00:00, 22.66it/s]

Test set: Average loss: 0.0012, Accuracy: 8043/10000 (80.43%)

EPOCH: 11
LR: 0.03391743770139635
Loss=1.3136634826660156 Batch_id=97 Accuracy=82.84: 100%|██████████| 98/98 [00:04<00:00, 23.09it/s]

Test set: Average loss: 0.0010, Accuracy: 8491/10000 (84.91%)

validation-accuracy improved from 82.71 to 84.91, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-11_L1-1_L2-0_val_acc-84.91.h5
EPOCH: 12
LR: 0.031525806122448985
Loss=1.2241768836975098 Batch_id=97 Accuracy=83.55: 100%|██████████| 98/98 [00:04<00:00, 22.33it/s]

Test set: Average loss: 0.0010, Accuracy: 8370/10000 (83.70%)

EPOCH: 13
LR: 0.029134174543501614
Loss=1.018428921699524 Batch_id=97 Accuracy=84.74: 100%|██████████| 98/98 [00:04<00:00, 23.04it/s]

Test set: Average loss: 0.0008, Accuracy: 8724/10000 (87.24%)

validation-accuracy improved from 84.91 to 87.24, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-13_L1-1_L2-0_val_acc-87.24.h5
EPOCH: 14
LR: 0.026742542964554243
Loss=1.0405094623565674 Batch_id=97 Accuracy=86.12: 100%|██████████| 98/98 [00:04<00:00, 22.10it/s]

Test set: Average loss: 0.0008, Accuracy: 8680/10000 (86.80%)

EPOCH: 15
LR: 0.024350911385606876
Loss=1.1151455640792847 Batch_id=97 Accuracy=85.86: 100%|██████████| 98/98 [00:04<00:00, 22.68it/s]

Test set: Average loss: 0.0009, Accuracy: 8533/10000 (85.33%)

EPOCH: 16
LR: 0.021959279806659508
Loss=1.043213129043579 Batch_id=97 Accuracy=86.62: 100%|██████████| 98/98 [00:04<00:00, 22.22it/s]

Test set: Average loss: 0.0008, Accuracy: 8596/10000 (85.96%)

EPOCH: 17
LR: 0.01956764822771214
Loss=0.9419251680374146 Batch_id=97 Accuracy=87.72: 100%|██████████| 98/98 [00:04<00:00, 22.51it/s]

Test set: Average loss: 0.0008, Accuracy: 8803/10000 (88.03%)

validation-accuracy improved from 87.24 to 88.03, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-17_L1-1_L2-0_val_acc-88.03.h5
EPOCH: 18
LR: 0.01717601664876477
Loss=0.9170505404472351 Batch_id=97 Accuracy=88.40: 100%|██████████| 98/98 [00:04<00:00, 22.95it/s]

Test set: Average loss: 0.0007, Accuracy: 8861/10000 (88.61%)

validation-accuracy improved from 88.03 to 88.61, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-18_L1-1_L2-0_val_acc-88.61.h5
EPOCH: 19
LR: 0.014784385069817399
Loss=0.9086548089981079 Batch_id=97 Accuracy=89.25: 100%|██████████| 98/98 [00:04<00:00, 22.83it/s]

Test set: Average loss: 0.0007, Accuracy: 8875/10000 (88.75%)

validation-accuracy improved from 88.61 to 88.75, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-19_L1-1_L2-0_val_acc-88.75.h5
EPOCH: 20
LR: 0.012392753490870032
Loss=0.8919439911842346 Batch_id=97 Accuracy=90.04: 100%|██████████| 98/98 [00:04<00:00, 22.89it/s]

Test set: Average loss: 0.0007, Accuracy: 8932/10000 (89.32%)

validation-accuracy improved from 88.75 to 89.32, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-20_L1-1_L2-0_val_acc-89.32.h5
EPOCH: 21
LR: 0.010001121911922664
Loss=0.8085275292396545 Batch_id=97 Accuracy=90.81: 100%|██████████| 98/98 [00:04<00:00, 22.43it/s]

Test set: Average loss: 0.0007, Accuracy: 8923/10000 (89.23%)

EPOCH: 22
LR: 0.007609490332975297
Loss=0.8469421863555908 Batch_id=97 Accuracy=91.55: 100%|██████████| 98/98 [00:04<00:00, 22.74it/s]

Test set: Average loss: 0.0006, Accuracy: 9051/10000 (90.51%)

validation-accuracy improved from 89.32 to 90.51, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-22_L1-1_L2-0_val_acc-90.51.h5
EPOCH: 23
LR: 0.005217858754027929
Loss=0.8245468735694885 Batch_id=97 Accuracy=92.60: 100%|██████████| 98/98 [00:04<00:00, 22.21it/s]

Test set: Average loss: 0.0006, Accuracy: 9128/10000 (91.28%)

validation-accuracy improved from 90.51 to 91.28, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-23_L1-1_L2-0_val_acc-91.28.h5
EPOCH: 24
LR: 0.0028262271750805618
Loss=0.7002050280570984 Batch_id=97 Accuracy=93.54: 100%|██████████| 98/98 [00:04<00:00, 22.98it/s]

Test set: Average loss: 0.0005, Accuracy: 9178/10000 (91.78%)

validation-accuracy improved from 91.28 to 91.78, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-24_L1-1_L2-0_val_acc-91.78.h5
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















