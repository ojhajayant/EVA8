
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

2.   Trained for EPOCHS= 24 epochs as required, the div_factor was taken as 10, so as to start the cycle with a learning rate of best_lr/10 = 0.003, it is required that the max LR is reached on 5th epoch, with NO annihilation epochs, hence final_div_factor = div_factor & MAX_LR_EPOCH = 5 thus resulting in PCT_START = MAX_LR_EPOCH / EPOCHS = 0.2

3.  max test/validation accuracy within 24 epochs =66.78%


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
        Rearrange-21            [-1, 64, 33, 1]               0
           Conv2d-22            [-1, 64, 33, 1]           4,096
             GELU-23            [-1, 64, 33, 1]               0
          Dropout-24            [-1, 64, 33, 1]               0
           Conv2d-25            [-1, 64, 33, 1]           4,096
          Dropout-26            [-1, 64, 33, 1]               0
        Rearrange-27               [-1, 33, 64]               0
      FeedForward-28               [-1, 33, 64]               0
          PreNorm-29               [-1, 33, 64]               0
        LayerNorm-30               [-1, 33, 64]             128
        Rearrange-31            [-1, 64, 33, 1]               0
           Conv2d-32           [-1, 192, 33, 1]          12,288
        Rearrange-33              [-1, 33, 192]               0
          Softmax-34            [-1, 4, 33, 33]               0
        Rearrange-35            [-1, 64, 33, 1]               0
           Conv2d-36            [-1, 64, 33, 1]           4,096
        Rearrange-37               [-1, 33, 64]               0
        Attention-38               [-1, 33, 64]               0
          PreNorm-39               [-1, 33, 64]               0
        LayerNorm-40               [-1, 33, 64]             128
        Rearrange-41            [-1, 64, 33, 1]               0
           Conv2d-42            [-1, 64, 33, 1]           4,096
             GELU-43            [-1, 64, 33, 1]               0
          Dropout-44            [-1, 64, 33, 1]               0
           Conv2d-45            [-1, 64, 33, 1]           4,096
          Dropout-46            [-1, 64, 33, 1]               0
        Rearrange-47               [-1, 33, 64]               0
      FeedForward-48               [-1, 33, 64]               0
          PreNorm-49               [-1, 33, 64]               0
      Transformer-50               [-1, 33, 64]               0
         Identity-51                   [-1, 64]               0
        LayerNorm-52                   [-1, 64]             128
        Rearrange-53             [-1, 64, 1, 1]               0
           Conv2d-54             [-1, 10, 1, 1]             650
        Rearrange-55                   [-1, 10]               0
================================================================
Total params: 71,914
Trainable params: 71,914
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 1.62
Params size (MB): 0.27
Estimated Total Size (MB): 1.91
----------------------------------------------------------------
```



Training Logs:

```
----------------------------------------------------------------
----------------------------------------------------------------
Model training starts on CIFAR10 dataset
EPOCH: 1
LR: 0.003
Loss=1.7556650638580322 Batch_id=97 Accuracy=28.43: 100%|██████████| 98/98 [00:06<00:00, 15.89it/s]
Test set: Average loss: 0.0032, Accuracy: 4132/10000 (41.32%)

validation-accuracy improved from 0 to 41.32, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-1_L1-1_L2-0_val_acc-41.32.h5
EPOCH: 2
LR: 0.008411042944785275
Loss=1.7271416187286377 Batch_id=97 Accuracy=40.66: 100%|██████████| 98/98 [00:05<00:00, 17.27it/s]
Test set: Average loss: 0.0029, Accuracy: 4657/10000 (46.57%)

validation-accuracy improved from 41.32 to 46.57, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-2_L1-1_L2-0_val_acc-46.57.h5
EPOCH: 3
LR: 0.01382208588957055
Loss=1.7119848728179932 Batch_id=97 Accuracy=43.20: 100%|██████████| 98/98 [00:05<00:00, 17.55it/s]
Test set: Average loss: 0.0029, Accuracy: 4641/10000 (46.41%)

EPOCH: 4
LR: 0.019233128834355826
Loss=1.652485966682434 Batch_id=97 Accuracy=44.32: 100%|██████████| 98/98 [00:05<00:00, 17.35it/s]
Test set: Average loss: 0.0029, Accuracy: 4857/10000 (48.57%)

validation-accuracy improved from 46.57 to 48.57, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-4_L1-1_L2-0_val_acc-48.57.h5
EPOCH: 5
LR: 0.024644171779141102
Loss=1.5584139823913574 Batch_id=97 Accuracy=44.18: 100%|██████████| 98/98 [00:05<00:00, 17.44it/s]
Test set: Average loss: 0.0027, Accuracy: 5116/10000 (51.16%)

validation-accuracy improved from 48.57 to 51.16, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-5_L1-1_L2-0_val_acc-51.16.h5
EPOCH: 6
LR: 0.02998404940923738
Loss=1.5917167663574219 Batch_id=97 Accuracy=45.73: 100%|██████████| 98/98 [00:05<00:00, 17.33it/s]
Test set: Average loss: 0.0029, Accuracy: 4782/10000 (47.82%)

EPOCH: 7
LR: 0.028420891514500536
Loss=1.6352331638336182 Batch_id=97 Accuracy=47.01: 100%|██████████| 98/98 [00:05<00:00, 16.92it/s]
Test set: Average loss: 0.0026, Accuracy: 5197/10000 (51.97%)

validation-accuracy improved from 51.16 to 51.97, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-7_L1-1_L2-0_val_acc-51.97.h5
EPOCH: 8
LR: 0.026857733619763693
Loss=1.5327650308609009 Batch_id=97 Accuracy=48.53: 100%|██████████| 98/98 [00:05<00:00, 17.03it/s]
Test set: Average loss: 0.0027, Accuracy: 5232/10000 (52.32%)

validation-accuracy improved from 51.97 to 52.32, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-8_L1-1_L2-0_val_acc-52.32.h5
EPOCH: 9
LR: 0.02529457572502685
Loss=1.5700397491455078 Batch_id=97 Accuracy=50.31: 100%|██████████| 98/98 [00:05<00:00, 17.54it/s]
Test set: Average loss: 0.0025, Accuracy: 5513/10000 (55.13%)

validation-accuracy improved from 52.32 to 55.13, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-9_L1-1_L2-0_val_acc-55.13.h5
EPOCH: 10
LR: 0.02373141783029001
Loss=1.511792540550232 Batch_id=97 Accuracy=50.74: 100%|██████████| 98/98 [00:05<00:00, 17.31it/s]
Test set: Average loss: 0.0024, Accuracy: 5609/10000 (56.09%)

validation-accuracy improved from 55.13 to 56.09, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-10_L1-1_L2-0_val_acc-56.09.h5
EPOCH: 11
LR: 0.022168259935553165
Loss=1.4840112924575806 Batch_id=97 Accuracy=51.93: 100%|██████████| 98/98 [00:05<00:00, 17.30it/s]
Test set: Average loss: 0.0024, Accuracy: 5612/10000 (56.12%)

validation-accuracy improved from 56.09 to 56.12, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-11_L1-1_L2-0_val_acc-56.12.h5
EPOCH: 12
LR: 0.020605102040816326
Loss=1.4405205249786377 Batch_id=97 Accuracy=52.44: 100%|██████████| 98/98 [00:05<00:00, 17.59it/s]
Test set: Average loss: 0.0024, Accuracy: 5694/10000 (56.94%)

validation-accuracy improved from 56.12 to 56.94, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-12_L1-1_L2-0_val_acc-56.94.h5
EPOCH: 13
LR: 0.019041944146079483
Loss=1.4468913078308105 Batch_id=97 Accuracy=53.69: 100%|██████████| 98/98 [00:05<00:00, 17.30it/s]
Test set: Average loss: 0.0023, Accuracy: 5888/10000 (58.88%)

validation-accuracy improved from 56.94 to 58.88, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-13_L1-1_L2-0_val_acc-58.88.h5
EPOCH: 14
LR: 0.017478786251342644
Loss=1.3786044120788574 Batch_id=97 Accuracy=54.91: 100%|██████████| 98/98 [00:05<00:00, 17.36it/s]
Test set: Average loss: 0.0022, Accuracy: 6009/10000 (60.09%)

validation-accuracy improved from 58.88 to 60.09, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-14_L1-1_L2-0_val_acc-60.09.h5
EPOCH: 15
LR: 0.0159156283566058
Loss=1.3809154033660889 Batch_id=97 Accuracy=55.58: 100%|██████████| 98/98 [00:05<00:00, 17.39it/s]
Test set: Average loss: 0.0022, Accuracy: 6098/10000 (60.98%)

validation-accuracy improved from 60.09 to 60.98, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-15_L1-1_L2-0_val_acc-60.98.h5
EPOCH: 16
LR: 0.014352470461868959
Loss=1.3026299476623535 Batch_id=97 Accuracy=56.31: 100%|██████████| 98/98 [00:05<00:00, 17.49it/s]
Test set: Average loss: 0.0021, Accuracy: 6182/10000 (61.82%)

validation-accuracy improved from 60.98 to 61.82, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-16_L1-1_L2-0_val_acc-61.82.h5
EPOCH: 17
LR: 0.01278931256713212
Loss=1.2449743747711182 Batch_id=97 Accuracy=57.38: 100%|██████████| 98/98 [00:05<00:00, 17.08it/s]
Test set: Average loss: 0.0021, Accuracy: 6086/10000 (60.86%)

EPOCH: 18
LR: 0.011226154672395273
Loss=1.28666090965271 Batch_id=97 Accuracy=57.65: 100%|██████████| 98/98 [00:05<00:00, 17.48it/s]
Test set: Average loss: 0.0022, Accuracy: 6055/10000 (60.55%)

EPOCH: 19
LR: 0.00966299677765843
Loss=1.2336550951004028 Batch_id=97 Accuracy=58.61: 100%|██████████| 98/98 [00:05<00:00, 16.97it/s]
Test set: Average loss: 0.0020, Accuracy: 6298/10000 (62.98%)

validation-accuracy improved from 61.82 to 62.98, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-19_L1-1_L2-0_val_acc-62.98.h5
EPOCH: 20
LR: 0.008099838882921592
Loss=1.1878244876861572 Batch_id=97 Accuracy=59.38: 100%|██████████| 98/98 [00:05<00:00, 17.14it/s]
Test set: Average loss: 0.0020, Accuracy: 6476/10000 (64.76%)

validation-accuracy improved from 62.98 to 64.76, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-20_L1-1_L2-0_val_acc-64.76.h5
EPOCH: 21
LR: 0.006536680988184749
Loss=1.213541865348816 Batch_id=97 Accuracy=60.41: 100%|██████████| 98/98 [00:05<00:00, 17.13it/s]
Test set: Average loss: 0.0020, Accuracy: 6444/10000 (64.44%)

EPOCH: 22
LR: 0.004973523093447906
Loss=1.229129433631897 Batch_id=97 Accuracy=61.00: 100%|██████████| 98/98 [00:05<00:00, 16.83it/s]
Test set: Average loss: 0.0019, Accuracy: 6613/10000 (66.13%)

validation-accuracy improved from 64.76 to 66.13, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-22_L1-1_L2-0_val_acc-66.13.h5
EPOCH: 23
LR: 0.0034103651987110635
Loss=1.2032792568206787 Batch_id=97 Accuracy=61.70: 100%|██████████| 98/98 [00:05<00:00, 17.25it/s]
Test set: Average loss: 0.0019, Accuracy: 6617/10000 (66.17%)

validation-accuracy improved from 66.13 to 66.17, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-23_L1-1_L2-0_val_acc-66.17.h5
EPOCH: 24
LR: 0.0018472073039742243
Loss=1.0314149856567383 Batch_id=97 Accuracy=62.46: 100%|██████████| 98/98 [00:05<00:00, 17.05it/s]
Test set: Average loss: 0.0018, Accuracy: 6678/10000 (66.78%)

validation-accuracy improved from 66.17 to 66.78, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-24_L1-1_L2-0_val_acc-66.78.h5
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











