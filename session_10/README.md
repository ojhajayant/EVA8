
# Session_10

### 

## Check out this [network](https://github.com/kentaroy47/vision-transformers-cifar10/blob/main/models/vit.py)

* Re-write this network such that it is similar to the network we wrote in the class

* All parameters are the same as the network we wrote

* Proceed to submit the assignment:

    > Share the model code and link to the model cost

    > Share the training logs

    > Share the gradcam images for 10 misclassified images




1.   [EVA8_API](https://github.com/ojhajayant/EVA8_API) is the main repo which is being cloned here to be able to run the main.py script with various user provided (or default) arg options.

2.   The LR-Range test over 500 epochs, as required for the One Cycle Policy (OCP) gave a max_lr(best_lr) of 0.022000000097799996, I am using a LR which is a little ahead in terms of rounding off (and also due to the accuracy Vs learning rate plot obtained from the LR-range test run below in this notebook)

3.   Trained for EPOCHS= 24 epochs as required, the div_factor was taken as 10, so as to start the cycle with a learning rate of best_lr/10 = 0.003, it is required that the max LR is reached on 5th epoch, with NO annihilation epochs, hence final_div_factor = div_factor & MAX_LR_EPOCH = 5 thus resulting in PCT_START = MAX_LR_EPOCH / EPOCHS = 0.2

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

	> init_lr : 1e-10

	> end_lr : 1

	> max_lr_epochs : 5

	> lr_range_test_epochs : 500

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

	> prefix : data

	> best_model : CIFAR10_model_epoch-24_L1-1_L2-0_val_acc-89.17.h5


5.  max test/validation accuracy within 24 epochs = 89.17%








Please refer the [notebook](https://github.com/ojhajayant/EVA8/blob/main/session_10/EVA8_session10_assignment.ipynb) for this assignment solution.



```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             896
              GELU-2           [-1, 32, 32, 32]               0
            Conv2d-3           [-1, 32, 16, 16]           4,128
              GELU-4           [-1, 32, 16, 16]               0
         Rearrange-5              [-1, 32, 256]               0
         Rearrange-6           [-1, 256, 32, 1]               0
            Conv2d-7            [-1, 64, 32, 1]          16,448
         Rearrange-8               [-1, 32, 64]               0
           Dropout-9               [-1, 33, 64]               0
        LayerNorm-10               [-1, 33, 64]             128
        Rearrange-11            [-1, 64, 33, 1]               0
           Conv2d-12           [-1, 192, 33, 1]          12,288
        Rearrange-13              [-1, 33, 192]               0
          Softmax-14            [-1, 4, 33, 33]               0
        Rearrange-15            [-1, 64, 33, 1]               0
           Conv2d-16            [-1, 64, 33, 1]           4,096
        Rearrange-17               [-1, 33, 64]               0
        Attention-18               [-1, 33, 64]               0
          PreNorm-19               [-1, 33, 64]               0
        LayerNorm-20               [-1, 33, 64]             128
           Linear-21               [-1, 33, 64]           4,160
             GELU-22               [-1, 33, 64]               0
          Dropout-23               [-1, 33, 64]               0
           Linear-24               [-1, 33, 64]           4,160
          Dropout-25               [-1, 33, 64]               0
      FeedForward-26               [-1, 33, 64]               0
          PreNorm-27               [-1, 33, 64]               0
        LayerNorm-28               [-1, 33, 64]             128
        Rearrange-29            [-1, 64, 33, 1]               0
           Conv2d-30           [-1, 192, 33, 1]          12,288
        Rearrange-31              [-1, 33, 192]               0
          Softmax-32            [-1, 4, 33, 33]               0
        Rearrange-33            [-1, 64, 33, 1]               0
           Conv2d-34            [-1, 64, 33, 1]           4,096
        Rearrange-35               [-1, 33, 64]               0
        Attention-36               [-1, 33, 64]               0
          PreNorm-37               [-1, 33, 64]               0
        LayerNorm-38               [-1, 33, 64]             128
           Linear-39               [-1, 33, 64]           4,160
             GELU-40               [-1, 33, 64]               0
          Dropout-41               [-1, 33, 64]               0
           Linear-42               [-1, 33, 64]           4,160
          Dropout-43               [-1, 33, 64]               0
      FeedForward-44               [-1, 33, 64]               0
          PreNorm-45               [-1, 33, 64]               0
      Transformer-46               [-1, 33, 64]               0
         Identity-47                   [-1, 64]               0
        LayerNorm-48                   [-1, 64]             128
        Rearrange-49             [-1, 64, 1, 1]               0
           Conv2d-50             [-1, 10, 1, 1]             650
        Rearrange-51                   [-1, 10]               0
================================================================
Total params: 72,170
Trainable params: 72,170
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 1.56
Params size (MB): 0.28
Estimated Total Size (MB): 1.85
----------------------------------------------------------------
```



Training Logs:

```
----------------------------------------------------------------
Model training starts on CIFAR10 dataset
EPOCH: 1
LR: 0.003
Loss=1.9587918519973755 Batch_id=97 Accuracy=27.70: 100%|██████████| 98/98 [00:06<00:00, 16.25it/s]
Test set: Average loss: 0.0033, Accuracy: 4084/10000 (40.84%)

validation-accuracy improved from 0 to 40.84, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-1_L1-1_L2-0_val_acc-40.84.h5
EPOCH: 2
LR: 0.008411042944785275
Loss=1.7193536758422852 Batch_id=97 Accuracy=40.73: 100%|██████████| 98/98 [00:05<00:00, 17.79it/s]
Test set: Average loss: 0.0031, Accuracy: 4515/10000 (45.15%)

validation-accuracy improved from 40.84 to 45.15, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-2_L1-1_L2-0_val_acc-45.15.h5
EPOCH: 3
LR: 0.01382208588957055
Loss=1.7354938983917236 Batch_id=97 Accuracy=43.16: 100%|██████████| 98/98 [00:05<00:00, 17.81it/s]
Test set: Average loss: 0.0027, Accuracy: 5099/10000 (50.99%)

validation-accuracy improved from 45.15 to 50.99, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-3_L1-1_L2-0_val_acc-50.99.h5
EPOCH: 4
LR: 0.019233128834355826
Loss=1.6892670392990112 Batch_id=97 Accuracy=43.82: 100%|██████████| 98/98 [00:05<00:00, 17.98it/s]
Test set: Average loss: 0.0028, Accuracy: 4970/10000 (49.70%)

EPOCH: 5
LR: 0.024644171779141102
Loss=1.6961021423339844 Batch_id=97 Accuracy=43.93: 100%|██████████| 98/98 [00:05<00:00, 17.49it/s]
Test set: Average loss: 0.0028, Accuracy: 5030/10000 (50.30%)

EPOCH: 6
LR: 0.02998404940923738
Loss=1.640432357788086 Batch_id=97 Accuracy=45.96: 100%|██████████| 98/98 [00:05<00:00, 18.13it/s]
Test set: Average loss: 0.0027, Accuracy: 5102/10000 (51.02%)

validation-accuracy improved from 50.99 to 51.02, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-6_L1-1_L2-0_val_acc-51.02.h5
EPOCH: 7
LR: 0.028420891514500536
Loss=1.4598321914672852 Batch_id=97 Accuracy=47.30: 100%|██████████| 98/98 [00:05<00:00, 17.85it/s]
Test set: Average loss: 0.0025, Accuracy: 5440/10000 (54.40%)

validation-accuracy improved from 51.02 to 54.4, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-7_L1-1_L2-0_val_acc-54.4.h5
EPOCH: 8
LR: 0.026857733619763693
Loss=1.532556176185608 Batch_id=97 Accuracy=48.91: 100%|██████████| 98/98 [00:05<00:00, 17.94it/s]
Test set: Average loss: 0.0027, Accuracy: 5182/10000 (51.82%)

EPOCH: 9
LR: 0.02529457572502685
Loss=1.468367576599121 Batch_id=97 Accuracy=49.88: 100%|██████████| 98/98 [00:05<00:00, 17.86it/s]
Test set: Average loss: 0.0024, Accuracy: 5678/10000 (56.78%)

validation-accuracy improved from 54.4 to 56.78, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-9_L1-1_L2-0_val_acc-56.78.h5
EPOCH: 10
LR: 0.02373141783029001
Loss=1.5203700065612793 Batch_id=97 Accuracy=51.97: 100%|██████████| 98/98 [00:05<00:00, 17.43it/s]
Test set: Average loss: 0.0024, Accuracy: 5670/10000 (56.70%)

EPOCH: 11
LR: 0.022168259935553165
Loss=1.4884910583496094 Batch_id=97 Accuracy=52.31: 100%|██████████| 98/98 [00:05<00:00, 17.18it/s]
Test set: Average loss: 0.0023, Accuracy: 5754/10000 (57.54%)

validation-accuracy improved from 56.78 to 57.54, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-11_L1-1_L2-0_val_acc-57.54.h5
EPOCH: 12
LR: 0.020605102040816326
Loss=1.3453963994979858 Batch_id=97 Accuracy=53.84: 100%|██████████| 98/98 [00:05<00:00, 17.84it/s]
Test set: Average loss: 0.0022, Accuracy: 6016/10000 (60.16%)

validation-accuracy improved from 57.54 to 60.16, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-12_L1-1_L2-0_val_acc-60.16.h5
EPOCH: 13
LR: 0.019041944146079483
Loss=1.3861701488494873 Batch_id=97 Accuracy=54.67: 100%|██████████| 98/98 [00:05<00:00, 17.86it/s]
Test set: Average loss: 0.0022, Accuracy: 6008/10000 (60.08%)

EPOCH: 14
LR: 0.017478786251342644
Loss=1.2919758558273315 Batch_id=97 Accuracy=55.54: 100%|██████████| 98/98 [00:05<00:00, 17.52it/s]
Test set: Average loss: 0.0021, Accuracy: 6166/10000 (61.66%)

validation-accuracy improved from 60.16 to 61.66, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-14_L1-1_L2-0_val_acc-61.66.h5
EPOCH: 15
LR: 0.0159156283566058
Loss=1.4092254638671875 Batch_id=97 Accuracy=56.29: 100%|██████████| 98/98 [00:05<00:00, 17.80it/s]
Test set: Average loss: 0.0022, Accuracy: 6089/10000 (60.89%)

EPOCH: 16
LR: 0.014352470461868959
Loss=1.3275909423828125 Batch_id=97 Accuracy=57.03: 100%|██████████| 98/98 [00:05<00:00, 17.83it/s]
Test set: Average loss: 0.0021, Accuracy: 6116/10000 (61.16%)

EPOCH: 17
LR: 0.01278931256713212
Loss=1.290686845779419 Batch_id=97 Accuracy=58.01: 100%|██████████| 98/98 [00:05<00:00, 18.14it/s]
Test set: Average loss: 0.0021, Accuracy: 6154/10000 (61.54%)

EPOCH: 18
LR: 0.011226154672395273
Loss=1.1936556100845337 Batch_id=97 Accuracy=58.97: 100%|██████████| 98/98 [00:05<00:00, 17.61it/s]
Test set: Average loss: 0.0021, Accuracy: 6270/10000 (62.70%)

validation-accuracy improved from 61.66 to 62.7, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-18_L1-1_L2-0_val_acc-62.7.h5
EPOCH: 19
LR: 0.00966299677765843
Loss=1.2783204317092896 Batch_id=97 Accuracy=58.96: 100%|██████████| 98/98 [00:05<00:00, 18.18it/s]
Test set: Average loss: 0.0020, Accuracy: 6441/10000 (64.41%)

validation-accuracy improved from 62.7 to 64.41, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-19_L1-1_L2-0_val_acc-64.41.h5
EPOCH: 20
LR: 0.008099838882921592
Loss=1.2840720415115356 Batch_id=97 Accuracy=60.16: 100%|██████████| 98/98 [00:05<00:00, 17.62it/s]
Test set: Average loss: 0.0019, Accuracy: 6494/10000 (64.94%)

validation-accuracy improved from 64.41 to 64.94, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-20_L1-1_L2-0_val_acc-64.94.h5
EPOCH: 21
LR: 0.006536680988184749
Loss=1.2337315082550049 Batch_id=97 Accuracy=61.09: 100%|██████████| 98/98 [00:05<00:00, 17.64it/s]
Test set: Average loss: 0.0019, Accuracy: 6586/10000 (65.86%)

validation-accuracy improved from 64.94 to 65.86, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-21_L1-1_L2-0_val_acc-65.86.h5
EPOCH: 22
LR: 0.004973523093447906
Loss=1.2584549188613892 Batch_id=97 Accuracy=62.10: 100%|██████████| 98/98 [00:05<00:00, 17.30it/s]
Test set: Average loss: 0.0019, Accuracy: 6620/10000 (66.20%)

validation-accuracy improved from 65.86 to 66.2, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-22_L1-1_L2-0_val_acc-66.2.h5
EPOCH: 23
LR: 0.0034103651987110635
Loss=1.130612850189209 Batch_id=97 Accuracy=62.76: 100%|██████████| 98/98 [00:05<00:00, 17.90it/s]
Test set: Average loss: 0.0018, Accuracy: 6653/10000 (66.53%)

validation-accuracy improved from 66.2 to 66.53, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-23_L1-1_L2-0_val_acc-66.53.h5
EPOCH: 24
LR: 0.0018472073039742243
Loss=1.144034504890442 Batch_id=97 Accuracy=63.49: 100%|██████████| 98/98 [00:05<00:00, 17.61it/s]
Test set: Average loss: 0.0018, Accuracy: 6758/10000 (67.58%)

validation-accuracy improved from 66.53 to 67.58, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-24_L1-1_L2-0_val_acc-67.58.h5
```


#### Few details on the Model:

Please refer [vit.py](https://github.com/ojhajayant/EVA8_API/blob/main/models/vit.py)
All the model files now reside in the same location like this vit.py file

```python
!git clone https://git@github.com/ojhajayant//EVA8_API.git
 ```


###  Following shows the training/validation loss plot:

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_10/train_test_loss_plt.png "Logo Title Text 1")

### Misclassified Images:

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_10/misclassified_images.png "Logo Title Text 1")

###  Confusion Matrix & Classification Reports:

For Best saved Model:

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_10/confusion_matrix.png "Logo Title Text 1")











