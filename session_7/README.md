
# Session_7

### 

> Check this Repo out: https://github.com/kuangliu/pytorch-cifar Links to an external site. 

> You are going to follow the same structure for your Code from now on. So 

 *   Create:
models folder - this is where you'll add all of your future models. 

*   Copy resnet.py into this folder, this file should only have ResNet 18/34 models. Delete Bottleneck Class

*   main.py - from Google Colab, now onwards, this is the file that you'll import (along with the model). Your main file shall be able to take these params or you should be able to pull functions from it and then perform operations, like (including but not limited to):

    > training and test loops

    > data split between test and train

    > epochs

    > batch size

    > which optimizer to run

    > do we run a scheduler?

*   utils.py file (or a folder later on when it expands) - this is where you will add all of your utilities like:

    > image transforms,

    > gradcam,

    > misclassification code,

    > tensorboard related stuff

    > advanced training policies, etc etc


*   Name this main repos something, and don't call it Assignment 7. This is what you'll import for all the rest of the assignments. Add a proper readme describing all the files. 

*   Your assignment is to build the above training structure. Train ResNet18 on Cifar10 for 20 Epochs. The assignment must:

    > pull your Github code to google colab (don't copy-paste code)

    > prove that you are following the above structure

    > that the code in your google collab notebook is NOTHING.. barely anything. There should not be any function or class that you can define in your Google Colab Notebook. Everything must be imported from all of your other files

*  your colab file must:

    > train resnet18 for 20 epochs on the CIFAR10 dataset

    > show loss curves for test and train datasets

    > show a gallery of 10 misclassified images

    > show gradcam Links to an external site.output on 10 misclassified images. Remember if you are applying GradCAM on a channel that is less than 5px, then please don't bother to submit the assignment. 😡🤬🤬🤬🤬

    > Once done, upload the code to GitHub, and share the code. This readme must link to the main repo so we can read your file structure. 


1.   [EVA8_API](https://github.com/ojhajayant/EVA8_API) is the main repo which is being clone here to be able to run the main.py script with various user provided (or default) arg options.

2.   The LR-Range test over 100 epochs, as required for the One Cycle Policy (OCP) gave a max_lr(best_lr) of 0.050499

3.   Trained for 20 epochs as required, the div_factor was taken as 10, so as to start the cycle with a learning rate of best_lr/10 = 0.0050499, wanetd to have same number of epochs for taking from min_lr to max_lr and vice-versa, with NO annihilation epochs, hence final_div_factor = div_factor & MAX_LR_EPOCH = EPOCHS // 2 thus resulting in PCT_START = MAX_LR_EPOCH / EPOCHS = 0.5

4.  Here are the different args values for this run:

	> cmd : Either of "lr_find", "train", "test"

	> IPYNB_ENV : True

	> use_albumentations : True

	> SEED : 1

	> dataset : CIFAR10

	> img_size : (32, 32)

	> batch_size : 128
  
        > epochs : 20

	> criterion : NLLLoss()

	> init_lr : 0.0001 (for LR-Range test)

	> end_lr : 0.05 (for LR-Range test)

	> lr_range_test_epochs : 100 (epochs used for LR-Range test)

	> best_lr : 0.504999999999

	> cycle_momentum : True

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

	> best_model :  CIFAR10_model_epoch-20_L1-1_L2-0_val_acc-90.6.h5


Please refer the [notebook](https://github.com/ojhajayant/EVA8/blob/main/session_7/EVA8_session7_assignment.ipynb) for this assignment solution.

> please note that under this "script mode" the plots could not be inlined/embedded into the colab cell, hence had to be saved as *.png files, the places in the code where the plots are supposed to be inlined appear as:
Figure(800x300)
Figure(800x300)

or similar, hence plotted/displayed the *png plot files separately in separate cell.
Here is some evidence snapshot on the plots generated during run time (which unfortunately couldn't come up as "embedded/inlined" in the notebook under this "script" mode):

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_7/sample_plot_saved_colab.png "Logo Title Text 1")

- max test/validation accuracy within 20 epochs = 90.60%

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
            Conv2d-3           [-1, 64, 32, 32]          36,864
       BatchNorm2d-4           [-1, 64, 32, 32]             128
            Conv2d-5           [-1, 64, 32, 32]          36,864
       BatchNorm2d-6           [-1, 64, 32, 32]             128
        BasicBlock-7           [-1, 64, 32, 32]               0
            Conv2d-8           [-1, 64, 32, 32]          36,864
       BatchNorm2d-9           [-1, 64, 32, 32]             128
           Conv2d-10           [-1, 64, 32, 32]          36,864
      BatchNorm2d-11           [-1, 64, 32, 32]             128
       BasicBlock-12           [-1, 64, 32, 32]               0
           Conv2d-13          [-1, 128, 16, 16]          73,728
      BatchNorm2d-14          [-1, 128, 16, 16]             256
           Conv2d-15          [-1, 128, 16, 16]         147,456
      BatchNorm2d-16          [-1, 128, 16, 16]             256
           Conv2d-17          [-1, 128, 16, 16]           8,192
      BatchNorm2d-18          [-1, 128, 16, 16]             256
       BasicBlock-19          [-1, 128, 16, 16]               0
           Conv2d-20          [-1, 128, 16, 16]         147,456
      BatchNorm2d-21          [-1, 128, 16, 16]             256
           Conv2d-22          [-1, 128, 16, 16]         147,456
      BatchNorm2d-23          [-1, 128, 16, 16]             256
       BasicBlock-24          [-1, 128, 16, 16]               0
           Conv2d-25            [-1, 256, 8, 8]         294,912
      BatchNorm2d-26            [-1, 256, 8, 8]             512
           Conv2d-27            [-1, 256, 8, 8]         589,824
      BatchNorm2d-28            [-1, 256, 8, 8]             512
           Conv2d-29            [-1, 256, 8, 8]          32,768
      BatchNorm2d-30            [-1, 256, 8, 8]             512
       BasicBlock-31            [-1, 256, 8, 8]               0
           Conv2d-32            [-1, 256, 8, 8]         589,824
      BatchNorm2d-33            [-1, 256, 8, 8]             512
           Conv2d-34            [-1, 256, 8, 8]         589,824
      BatchNorm2d-35            [-1, 256, 8, 8]             512
       BasicBlock-36            [-1, 256, 8, 8]               0
           Conv2d-37            [-1, 512, 4, 4]       1,179,648
      BatchNorm2d-38            [-1, 512, 4, 4]           1,024
           Conv2d-39            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-40            [-1, 512, 4, 4]           1,024
           Conv2d-41            [-1, 512, 4, 4]         131,072
      BatchNorm2d-42            [-1, 512, 4, 4]           1,024
       BasicBlock-43            [-1, 512, 4, 4]               0
           Conv2d-44            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-45            [-1, 512, 4, 4]           1,024
           Conv2d-46            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-47            [-1, 512, 4, 4]           1,024
       BasicBlock-48            [-1, 512, 4, 4]               0
           Linear-49                   [-1, 10]           5,130
================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 11.25
Params size (MB): 42.63
Estimated Total Size (MB): 53.89
----------------------------------------------------------------
```

Logs:

```
Estimated Total Size (MB): 53.89
----------------------------------------------------------------
Model training starts on CIFAR10 dataset
EPOCH: 1
LR: 0.0050499
Loss=3.694809913635254 Batch_id=390 Accuracy=39.45: 100% 391/391 [00:09<00:00, 42.23it/s]

Test set: Average loss: 1.3296, Accuracy: 5320/10000 (53.20%)

validation-accuracy improved from 0 to 53.2, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-1_L1-1_L2-0_val_acc-53.2.h5
EPOCH: 2
LR: 0.009595972678434383
Loss=3.434737205505371 Batch_id=390 Accuracy=55.26: 100% 391/391 [00:09<00:00, 43.30it/s]

Test set: Average loss: 1.0956, Accuracy: 6144/10000 (61.44%)

validation-accuracy improved from 53.2 to 61.44, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-2_L1-1_L2-0_val_acc-61.44.h5
EPOCH: 3
LR: 0.014142045356868766
Loss=2.475287437438965 Batch_id=390 Accuracy=62.42: 100% 391/391 [00:09<00:00, 42.60it/s]

Test set: Average loss: 1.1301, Accuracy: 6340/10000 (63.40%)

validation-accuracy improved from 61.44 to 63.4, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-3_L1-1_L2-0_val_acc-63.4.h5
EPOCH: 4
LR: 0.01868811803530315
Loss=2.3686108589172363 Batch_id=390 Accuracy=66.22: 100% 391/391 [00:09<00:00, 42.55it/s]

Test set: Average loss: 0.9040, Accuracy: 6919/10000 (69.19%)

validation-accuracy improved from 63.4 to 69.19, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-4_L1-1_L2-0_val_acc-69.19.h5
EPOCH: 5
LR: 0.02323419071373753
Loss=1.7588059902191162 Batch_id=390 Accuracy=68.70: 100% 391/391 [00:08<00:00, 43.60it/s]

Test set: Average loss: 0.7598, Accuracy: 7495/10000 (74.95%)

validation-accuracy improved from 69.19 to 74.95, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-5_L1-1_L2-0_val_acc-74.95.h5
EPOCH: 6
LR: 0.02778026339217191
Loss=1.6672797203063965 Batch_id=390 Accuracy=70.17: 100% 391/391 [00:08<00:00, 43.64it/s]

Test set: Average loss: 0.7163, Accuracy: 7533/10000 (75.33%)

validation-accuracy improved from 74.95 to 75.33, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-6_L1-1_L2-0_val_acc-75.33.h5
EPOCH: 7
LR: 0.032326336070606296
Loss=1.701971173286438 Batch_id=390 Accuracy=71.39: 100% 391/391 [00:08<00:00, 43.76it/s]

Test set: Average loss: 0.8995, Accuracy: 7009/10000 (70.09%)

EPOCH: 8
LR: 0.03687240874904068
Loss=1.666421890258789 Batch_id=390 Accuracy=73.11: 100% 391/391 [00:09<00:00, 42.72it/s]

Test set: Average loss: 0.8559, Accuracy: 7210/10000 (72.10%)

EPOCH: 9
LR: 0.04141848142747506
Loss=1.6155893802642822 Batch_id=390 Accuracy=74.04: 100% 391/391 [00:08<00:00, 43.57it/s]

Test set: Average loss: 0.7592, Accuracy: 7464/10000 (74.64%)

EPOCH: 10
LR: 0.04596455410590944
Loss=1.5981957912445068 Batch_id=390 Accuracy=74.85: 100% 391/391 [00:08<00:00, 44.09it/s]

Test set: Average loss: 0.7683, Accuracy: 7574/10000 (75.74%)

validation-accuracy improved from 75.33 to 75.74, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-10_L1-1_L2-0_val_acc-75.74.h5
EPOCH: 11
LR: 0.05048621380818415
Loss=1.2647907733917236 Batch_id=390 Accuracy=75.63: 100% 391/391 [00:08<00:00, 43.53it/s]

Test set: Average loss: 0.9623, Accuracy: 6982/10000 (69.82%)

EPOCH: 12
LR: 0.045486812808184146
Loss=1.5850930213928223 Batch_id=390 Accuracy=76.53: 100% 391/391 [00:08<00:00, 44.15it/s]

Test set: Average loss: 0.6692, Accuracy: 7816/10000 (78.16%)

validation-accuracy improved from 75.74 to 78.16, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-12_L1-1_L2-0_val_acc-78.16.h5
EPOCH: 13
LR: 0.040487411808184146
Loss=1.3547148704528809 Batch_id=390 Accuracy=77.55: 100% 391/391 [00:09<00:00, 42.55it/s]

Test set: Average loss: 0.7759, Accuracy: 7864/10000 (78.64%)

validation-accuracy improved from 78.16 to 78.64, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-13_L1-1_L2-0_val_acc-78.64.h5
EPOCH: 14
LR: 0.035488010808184145
Loss=1.329831600189209 Batch_id=390 Accuracy=78.42: 100% 391/391 [00:09<00:00, 43.21it/s]

Test set: Average loss: 0.5671, Accuracy: 8080/10000 (80.80%)

validation-accuracy improved from 78.64 to 80.8, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-14_L1-1_L2-0_val_acc-80.8.h5
EPOCH: 15
LR: 0.030488609808184144
Loss=1.1353683471679688 Batch_id=390 Accuracy=79.27: 100% 391/391 [00:09<00:00, 42.77it/s]

Test set: Average loss: 0.4355, Accuracy: 8493/10000 (84.93%)

validation-accuracy improved from 80.8 to 84.93, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-15_L1-1_L2-0_val_acc-84.93.h5
EPOCH: 16
LR: 0.02548920880818414
Loss=1.1917486190795898 Batch_id=390 Accuracy=80.62: 100% 391/391 [00:09<00:00, 42.94it/s]

Test set: Average loss: 0.4438, Accuracy: 8482/10000 (84.82%)

EPOCH: 17
LR: 0.02048980780818414
Loss=1.208369255065918 Batch_id=390 Accuracy=80.96: 100% 391/391 [00:08<00:00, 44.17it/s]

Test set: Average loss: 0.5189, Accuracy: 8317/10000 (83.17%)

EPOCH: 18
LR: 0.015490406808184143
Loss=0.9628164768218994 Batch_id=390 Accuracy=82.10: 100% 391/391 [00:09<00:00, 43.40it/s]

Test set: Average loss: 0.3769, Accuracy: 8719/10000 (87.19%)

validation-accuracy improved from 84.93 to 87.19, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-18_L1-1_L2-0_val_acc-87.19.h5
EPOCH: 19
LR: 0.010491005808184142
Loss=1.135435938835144 Batch_id=390 Accuracy=83.93: 100% 391/391 [00:09<00:00, 40.65it/s]

Test set: Average loss: 0.3444, Accuracy: 8841/10000 (88.41%)

validation-accuracy improved from 87.19 to 88.41, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-19_L1-1_L2-0_val_acc-88.41.h5
EPOCH: 20
LR: 0.005491604808184142
Loss=0.8878072500228882 Batch_id=390 Accuracy=86.03: 100% 391/391 [00:09<00:00, 41.98it/s]

Test set: Average loss: 0.2747, Accuracy: 9060/10000 (90.60%)

validation-accuracy improved from 88.41 to 90.6, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-20_L1-1_L2-0_val_acc-90.6.h5

```




#### Few details on the Model:

Please refer [resnet.py](https://github.com/ojhajayant/EVA8_API/blob/main/models/resnet18.py)
All the model files now reside in the same location like this resnet.py file

```python
!git clone https://git@github.com/ojhajayant//EVA8_API.git
 ```


###  Following graph shows the model accuracy:

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_7/plot7.png "Logo Title Text 1")

###  Following graph shows the change in momentum & LR for the Once Cycle Policy across different iterations:

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_7/plot8.png "Logo Title Text 1")


###  Misclassified images:


For Best saved Model :

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_7/plot5.png "Logo Title Text 1")

With Grad Cam heatmap :

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_7/plot10.png "Logo Title Text 1")



###  Confusion Matrix & Classification Reports:

For Best saved Model:

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_7/confusion_matric_clsfcn_rpt.png "Logo Title Text 1")















