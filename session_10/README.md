
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

3.  max test/validation accuracy within 24 epochs = 67.52%


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
Loss=1.7554500102996826 Batch_id=97 Accuracy=28.42: 100%|██████████| 98/98 [00:06<00:00, 15.96it/s]
Test set: Average loss: 0.0032, Accuracy: 4131/10000 (41.31%)

validation-accuracy improved from 0 to 41.31, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-1_L1-1_L2-0_val_acc-41.31.h5
EPOCH: 2
LR: 0.008411042944785275
Loss=1.781583309173584 Batch_id=97 Accuracy=40.57: 100%|██████████| 98/98 [00:05<00:00, 17.83it/s]
Test set: Average loss: 0.0030, Accuracy: 4495/10000 (44.95%)

validation-accuracy improved from 41.31 to 44.95, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-2_L1-1_L2-0_val_acc-44.95.h5
EPOCH: 3
LR: 0.01382208588957055
Loss=1.672245740890503 Batch_id=97 Accuracy=43.81: 100%|██████████| 98/98 [00:05<00:00, 18.12it/s]
Test set: Average loss: 0.0029, Accuracy: 4674/10000 (46.74%)

validation-accuracy improved from 44.95 to 46.74, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-3_L1-1_L2-0_val_acc-46.74.h5
EPOCH: 4
LR: 0.019233128834355826
Loss=1.6134226322174072 Batch_id=97 Accuracy=44.20: 100%|██████████| 98/98 [00:05<00:00, 17.75it/s]
Test set: Average loss: 0.0027, Accuracy: 4986/10000 (49.86%)

validation-accuracy improved from 46.74 to 49.86, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-4_L1-1_L2-0_val_acc-49.86.h5
EPOCH: 5
LR: 0.024644171779141102
Loss=1.63678777217865 Batch_id=97 Accuracy=43.51: 100%|██████████| 98/98 [00:05<00:00, 18.17it/s]
Test set: Average loss: 0.0031, Accuracy: 4644/10000 (46.44%)

EPOCH: 6
LR: 0.02998404940923738
Loss=1.6279914379119873 Batch_id=97 Accuracy=45.43: 100%|██████████| 98/98 [00:05<00:00, 17.72it/s]
Test set: Average loss: 0.0028, Accuracy: 4891/10000 (48.91%)

EPOCH: 7
LR: 0.028420891514500536
Loss=1.5815659761428833 Batch_id=97 Accuracy=47.65: 100%|██████████| 98/98 [00:05<00:00, 18.13it/s]
Test set: Average loss: 0.0026, Accuracy: 5342/10000 (53.42%)

validation-accuracy improved from 49.86 to 53.42, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-7_L1-1_L2-0_val_acc-53.42.h5
EPOCH: 8
LR: 0.026857733619763693
Loss=1.5782020092010498 Batch_id=97 Accuracy=49.10: 100%|██████████| 98/98 [00:05<00:00, 17.75it/s]
Test set: Average loss: 0.0025, Accuracy: 5515/10000 (55.15%)

validation-accuracy improved from 53.42 to 55.15, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-8_L1-1_L2-0_val_acc-55.15.h5
EPOCH: 9
LR: 0.02529457572502685
Loss=1.5682899951934814 Batch_id=97 Accuracy=50.57: 100%|██████████| 98/98 [00:05<00:00, 17.89it/s]
Test set: Average loss: 0.0025, Accuracy: 5395/10000 (53.95%)

EPOCH: 10
LR: 0.02373141783029001
Loss=1.5825049877166748 Batch_id=97 Accuracy=50.88: 100%|██████████| 98/98 [00:05<00:00, 17.80it/s]
Test set: Average loss: 0.0024, Accuracy: 5632/10000 (56.32%)

validation-accuracy improved from 55.15 to 56.32, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-10_L1-1_L2-0_val_acc-56.32.h5
EPOCH: 11
LR: 0.022168259935553165
Loss=1.4603012800216675 Batch_id=97 Accuracy=52.52: 100%|██████████| 98/98 [00:05<00:00, 17.96it/s]
Test set: Average loss: 0.0024, Accuracy: 5733/10000 (57.33%)

validation-accuracy improved from 56.32 to 57.33, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-11_L1-1_L2-0_val_acc-57.33.h5
EPOCH: 12
LR: 0.020605102040816326
Loss=1.453799843788147 Batch_id=97 Accuracy=53.04: 100%|██████████| 98/98 [00:05<00:00, 18.25it/s]
Test set: Average loss: 0.0023, Accuracy: 5805/10000 (58.05%)

validation-accuracy improved from 57.33 to 58.05, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-12_L1-1_L2-0_val_acc-58.05.h5
EPOCH: 13
LR: 0.019041944146079483
Loss=1.4936953783035278 Batch_id=97 Accuracy=54.01: 100%|██████████| 98/98 [00:05<00:00, 17.75it/s]
Test set: Average loss: 0.0023, Accuracy: 5873/10000 (58.73%)

validation-accuracy improved from 58.05 to 58.73, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-13_L1-1_L2-0_val_acc-58.73.h5
EPOCH: 14
LR: 0.017478786251342644
Loss=1.3427090644836426 Batch_id=97 Accuracy=54.96: 100%|██████████| 98/98 [00:05<00:00, 18.14it/s]
Test set: Average loss: 0.0022, Accuracy: 5988/10000 (59.88%)

validation-accuracy improved from 58.73 to 59.88, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-14_L1-1_L2-0_val_acc-59.88.h5
EPOCH: 15
LR: 0.0159156283566058
Loss=1.4042139053344727 Batch_id=97 Accuracy=55.69: 100%|██████████| 98/98 [00:05<00:00, 17.69it/s]
Test set: Average loss: 0.0022, Accuracy: 6056/10000 (60.56%)

validation-accuracy improved from 59.88 to 60.56, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-15_L1-1_L2-0_val_acc-60.56.h5
EPOCH: 16
LR: 0.014352470461868959
Loss=1.2993903160095215 Batch_id=97 Accuracy=56.41: 100%|██████████| 98/98 [00:05<00:00, 17.94it/s]
Test set: Average loss: 0.0021, Accuracy: 6142/10000 (61.42%)

validation-accuracy improved from 60.56 to 61.42, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-16_L1-1_L2-0_val_acc-61.42.h5
EPOCH: 17
LR: 0.01278931256713212
Loss=1.2752257585525513 Batch_id=97 Accuracy=57.37: 100%|██████████| 98/98 [00:05<00:00, 17.45it/s]
Test set: Average loss: 0.0021, Accuracy: 6272/10000 (62.72%)

validation-accuracy improved from 61.42 to 62.72, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-17_L1-1_L2-0_val_acc-62.72.h5
EPOCH: 18
LR: 0.011226154672395273
Loss=1.2637327909469604 Batch_id=97 Accuracy=58.41: 100%|██████████| 98/98 [00:05<00:00, 17.70it/s]
Test set: Average loss: 0.0021, Accuracy: 6267/10000 (62.67%)

EPOCH: 19
LR: 0.00966299677765843
Loss=1.2177902460098267 Batch_id=97 Accuracy=59.31: 100%|██████████| 98/98 [00:05<00:00, 17.75it/s]
Test set: Average loss: 0.0020, Accuracy: 6349/10000 (63.49%)

validation-accuracy improved from 62.72 to 63.49, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-19_L1-1_L2-0_val_acc-63.49.h5
EPOCH: 20
LR: 0.008099838882921592
Loss=1.1729973554611206 Batch_id=97 Accuracy=59.91: 100%|██████████| 98/98 [00:05<00:00, 17.95it/s]
Test set: Average loss: 0.0019, Accuracy: 6518/10000 (65.18%)

validation-accuracy improved from 63.49 to 65.18, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-20_L1-1_L2-0_val_acc-65.18.h5
EPOCH: 21
LR: 0.006536680988184749
Loss=1.1627588272094727 Batch_id=97 Accuracy=60.74: 100%|██████████| 98/98 [00:05<00:00, 17.57it/s]
Test set: Average loss: 0.0020, Accuracy: 6461/10000 (64.61%)

EPOCH: 22
LR: 0.004973523093447906
Loss=1.204944372177124 Batch_id=97 Accuracy=61.91: 100%|██████████| 98/98 [00:05<00:00, 17.87it/s]
Test set: Average loss: 0.0019, Accuracy: 6606/10000 (66.06%)

validation-accuracy improved from 65.18 to 66.06, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-22_L1-1_L2-0_val_acc-66.06.h5
EPOCH: 23
LR: 0.0034103651987110635
Loss=1.2105414867401123 Batch_id=97 Accuracy=62.36: 100%|██████████| 98/98 [00:05<00:00, 17.66it/s]
Test set: Average loss: 0.0019, Accuracy: 6688/10000 (66.88%)

validation-accuracy improved from 66.06 to 66.88, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-23_L1-1_L2-0_val_acc-66.88.h5
EPOCH: 24
LR: 0.0018472073039742243
Loss=1.0605604648590088 Batch_id=97 Accuracy=63.38: 100%|██████████| 98/98 [00:05<00:00, 17.87it/s]
Test set: Average loss: 0.0018, Accuracy: 6752/10000 (67.52%)

validation-accuracy improved from 66.88 to 67.52, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-24_L1-1_L2-0_val_acc-67.52.h5
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











