
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
Model training starts on CIFAR10 dataset
EPOCH: 1
LR: 0.0207
Loss=2.405869483947754 Batch_id=97 Accuracy=19.18: 100%|██████████| 98/98 [00:05<00:00, 18.20it/s]

Test set: Average loss: 0.0041, Accuracy: 2268/10000 (22.68%)

validation-accuracy improved from 0 to 22.68, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-1_L1-1_L2-0_val_acc-22.68.h5
EPOCH: 2
LR: 0.05803619631901841
Loss=2.5853610038757324 Batch_id=97 Accuracy=19.60: 100%|██████████| 98/98 [00:05<00:00, 17.83it/s]

Test set: Average loss: 0.0045, Accuracy: 1487/10000 (14.87%)

EPOCH: 3
LR: 0.0953723926380368
Loss=2.5318238735198975 Batch_id=97 Accuracy=15.50: 100%|██████████| 98/98 [00:05<00:00, 18.00it/s]

Test set: Average loss: 0.0045, Accuracy: 1599/10000 (15.99%)

EPOCH: 4
LR: 0.1327085889570552
Loss=2.7241134643554688 Batch_id=97 Accuracy=12.60: 100%|██████████| 98/98 [00:05<00:00, 17.60it/s]

Test set: Average loss: 0.0046, Accuracy: 1064/10000 (10.64%)

EPOCH: 5
LR: 0.1700447852760736
Loss=2.661092519760132 Batch_id=97 Accuracy=13.27: 100%|██████████| 98/98 [00:05<00:00, 17.40it/s]

Test set: Average loss: 0.0047, Accuracy: 1018/10000 (10.18%)

EPOCH: 6
LR: 0.20688994092373791
Loss=2.770573854446411 Batch_id=97 Accuracy=12.85: 100%|██████████| 98/98 [00:05<00:00, 17.46it/s]

Test set: Average loss: 0.0047, Accuracy: 1161/10000 (11.61%)

EPOCH: 7
LR: 0.1961041514500537
Loss=2.728909969329834 Batch_id=97 Accuracy=12.73: 100%|██████████| 98/98 [00:05<00:00, 17.59it/s]

Test set: Average loss: 0.0046, Accuracy: 1278/10000 (12.78%)

EPOCH: 8
LR: 0.18531836197636947
Loss=2.6508164405822754 Batch_id=97 Accuracy=13.39: 100%|██████████| 98/98 [00:05<00:00, 17.91it/s]

Test set: Average loss: 0.0045, Accuracy: 1417/10000 (14.17%)

EPOCH: 9
LR: 0.17453257250268528
Loss=2.6197385787963867 Batch_id=97 Accuracy=13.40: 100%|██████████| 98/98 [00:05<00:00, 17.87it/s]

Test set: Average loss: 0.0045, Accuracy: 1480/10000 (14.80%)

EPOCH: 10
LR: 0.16374678302900106
Loss=2.580998420715332 Batch_id=97 Accuracy=13.12: 100%|██████████| 98/98 [00:05<00:00, 17.51it/s]

Test set: Average loss: 0.0046, Accuracy: 1398/10000 (13.98%)

EPOCH: 11
LR: 0.15296099355531684
Loss=2.573256015777588 Batch_id=97 Accuracy=12.81: 100%|██████████| 98/98 [00:05<00:00, 16.98it/s]

Test set: Average loss: 0.0046, Accuracy: 1333/10000 (13.33%)

EPOCH: 12
LR: 0.14217520408163264
Loss=2.488589286804199 Batch_id=97 Accuracy=12.76: 100%|██████████| 98/98 [00:05<00:00, 16.74it/s]

Test set: Average loss: 0.0045, Accuracy: 1511/10000 (15.11%)

EPOCH: 13
LR: 0.13138941460794842
Loss=2.6175787448883057 Batch_id=97 Accuracy=12.93: 100%|██████████| 98/98 [00:05<00:00, 16.72it/s]

Test set: Average loss: 0.0046, Accuracy: 1306/10000 (13.06%)

EPOCH: 14
LR: 0.12060362513426422
Loss=2.522526264190674 Batch_id=97 Accuracy=12.64: 100%|██████████| 98/98 [00:05<00:00, 17.13it/s]

Test set: Average loss: 0.0046, Accuracy: 1336/10000 (13.36%)

EPOCH: 15
LR: 0.10981783566058001
Loss=2.46802020072937 Batch_id=97 Accuracy=12.58: 100%|██████████| 98/98 [00:05<00:00, 17.01it/s]

Test set: Average loss: 0.0045, Accuracy: 1224/10000 (12.24%)

EPOCH: 16
LR: 0.0990320461868958
Loss=2.469705104827881 Batch_id=97 Accuracy=13.18: 100%|██████████| 98/98 [00:05<00:00, 17.22it/s]

Test set: Average loss: 0.0045, Accuracy: 1454/10000 (14.54%)

EPOCH: 17
LR: 0.0882462567132116
Loss=2.4956634044647217 Batch_id=97 Accuracy=12.97: 100%|██████████| 98/98 [00:05<00:00, 16.83it/s]

Test set: Average loss: 0.0045, Accuracy: 1406/10000 (14.06%)

EPOCH: 18
LR: 0.07746046723952738
Loss=2.4935874938964844 Batch_id=97 Accuracy=13.03: 100%|██████████| 98/98 [00:05<00:00, 16.70it/s]

Test set: Average loss: 0.0045, Accuracy: 1532/10000 (15.32%)

EPOCH: 19
LR: 0.06667467776584315
Loss=2.4721839427948 Batch_id=97 Accuracy=14.32: 100%|██████████| 98/98 [00:05<00:00, 16.99it/s]

Test set: Average loss: 0.0045, Accuracy: 1604/10000 (16.04%)

EPOCH: 20
LR: 0.05588888829215896
Loss=2.500047206878662 Batch_id=97 Accuracy=14.77: 100%|██████████| 98/98 [00:05<00:00, 17.09it/s]

Test set: Average loss: 0.0045, Accuracy: 1700/10000 (17.00%)

EPOCH: 21
LR: 0.04510309881847474
Loss=2.542085647583008 Batch_id=97 Accuracy=14.87: 100%|██████████| 98/98 [00:05<00:00, 16.82it/s]

Test set: Average loss: 0.0045, Accuracy: 1531/10000 (15.31%)

EPOCH: 22
LR: 0.03431730934479055
Loss=2.539330244064331 Batch_id=97 Accuracy=14.83: 100%|██████████| 98/98 [00:05<00:00, 16.97it/s]

Test set: Average loss: 0.0045, Accuracy: 1515/10000 (15.15%)

EPOCH: 23
LR: 0.023531519871106327
Loss=2.556309700012207 Batch_id=97 Accuracy=14.89: 100%|██████████| 98/98 [00:05<00:00, 16.98it/s]

Test set: Average loss: 0.0045, Accuracy: 1486/10000 (14.86%)

EPOCH: 24
LR: 0.012745730397422106
Loss=2.537597417831421 Batch_id=97 Accuracy=15.31: 100%|██████████| 98/98 [00:05<00:00, 17.08it/s]

Test set: Average loss: 0.0045, Accuracy: 1616/10000 (16.16%)
```


#### Few details on the Model:

Please refer [vit.py](https://github.com/ojhajayant/EVA8_API/blob/main/models/vit.py)
All the model files now reside in the same location like this vit.py file

```python
!git clone https://git@github.com/ojhajayant//EVA8_API.git
 ```


###  Following graph shows the model accuracy:

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_10/model_acc.png "Logo Title Text 1")













