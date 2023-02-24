
# Session_9

### 

# Build the following network:

* That takes a CIFAR10 image (32x32x3)

  > Add 3 Convolutions to arrive at AxAx48 dimensions (e.g. 32x32x3 | 3x3x3x16 >> 3x3x16x32 >> 3x3x32x48)

  > Apply GAP and get 1x1x48, call this X

* Create a block called ULTIMUS that:

  > Creates 3 FC layers called K, Q and V such that:

         X*K = 48*48x8 > 8

         X*Q = 48*48x8 > 8 

         X*V = 48*48x8 > 8 

         then create AM = SoftMax(QTK)/(8^0.5) = 8*8 = 8

         then Z = V*AM = 8*8 > 8

         then another FC layer called Out that:

         Z*Out = 8*8x48 > 48

* Repeat this Ultimus block 4 times

* Then add final FC layer that converts 48 to 10 and sends it to the loss function.

* Model would look like this C>C>C>U>U>U>U>FFC>Loss

* Train the model for 24 epochs using the OCP, Use ADAM as an optimizer. 

* FC Layer 


* SoftMax





1.   [EVA8_API](https://github.com/ojhajayant/EVA8_API) is the main repo which is being cloned here to be able to run the main.py script with various user provided (or default) arg options.



2.  Here are the different args values for this run:

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
	
	> best_lr : 0.151
	
	> cycle_momentum : True
	
	> div_factor : 10
	
	> optimizer : <class 'torch.optim.adam.Adam'>
	
	> cuda : True
	
	> dropout : 0.08
	
	> l1_weight : 2.5e-05
	
	> l2_weight_decay : 0.0002125
	
	> L1 : True
	
	> L2 : False
	
	> data : ./data/
	
	> best_model_path : ./saved_models/
	
	> prefix : data
	
	> best_model : CIFAR10_model_epoch-24_L1-1_L2-0_val_acc-41.1.h5


Please refer the [notebook](https://github.com/ojhajayant/EVA8/blob/main/session_9/EVA8_session9_assignment.ipynb) for this assignment solution.



```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 32, 32]             432
       BatchNorm2d-2           [-1, 16, 32, 32]              32
           Dropout-3           [-1, 16, 32, 32]               0
              ReLU-4           [-1, 16, 32, 32]               0
            Conv2d-5           [-1, 32, 32, 32]           4,608
       BatchNorm2d-6           [-1, 32, 32, 32]              64
           Dropout-7           [-1, 32, 32, 32]               0
              ReLU-8           [-1, 32, 32, 32]               0
            Conv2d-9           [-1, 48, 32, 32]          13,824
      BatchNorm2d-10           [-1, 48, 32, 32]              96
          Dropout-11           [-1, 48, 32, 32]               0
             ReLU-12           [-1, 48, 32, 32]               0
        AvgPool2d-13             [-1, 48, 1, 1]               0
           Linear-14                    [-1, 8]             392
          Dropout-15                    [-1, 8]               0
             ReLU-16                    [-1, 8]               0
           Linear-17                    [-1, 8]             392
          Dropout-18                    [-1, 8]               0
             ReLU-19                    [-1, 8]               0
           Linear-20                    [-1, 8]             392
          Dropout-21                    [-1, 8]               0
             ReLU-22                    [-1, 8]               0
           Linear-23                   [-1, 48]             432
          Dropout-24                   [-1, 48]               0
             ReLU-25                   [-1, 48]               0
          ULTIMUS-26                   [-1, 48]               0
           Linear-27                    [-1, 8]             392
          Dropout-28                    [-1, 8]               0
             ReLU-29                    [-1, 8]               0
           Linear-30                    [-1, 8]             392
          Dropout-31                    [-1, 8]               0
             ReLU-32                    [-1, 8]               0
           Linear-33                    [-1, 8]             392
          Dropout-34                    [-1, 8]               0
             ReLU-35                    [-1, 8]               0
           Linear-36                   [-1, 48]             432
          Dropout-37                   [-1, 48]               0
             ReLU-38                   [-1, 48]               0
          ULTIMUS-39                   [-1, 48]               0
           Linear-40                    [-1, 8]             392
          Dropout-41                    [-1, 8]               0
             ReLU-42                    [-1, 8]               0
           Linear-43                    [-1, 8]             392
          Dropout-44                    [-1, 8]               0
             ReLU-45                    [-1, 8]               0
           Linear-46                    [-1, 8]             392
          Dropout-47                    [-1, 8]               0
             ReLU-48                    [-1, 8]               0
           Linear-49                   [-1, 48]             432
          Dropout-50                   [-1, 48]               0
             ReLU-51                   [-1, 48]               0
          ULTIMUS-52                   [-1, 48]               0
           Linear-53                    [-1, 8]             392
          Dropout-54                    [-1, 8]               0
             ReLU-55                    [-1, 8]               0
           Linear-56                    [-1, 8]             392
          Dropout-57                    [-1, 8]               0
             ReLU-58                    [-1, 8]               0
           Linear-59                    [-1, 8]             392
          Dropout-60                    [-1, 8]               0
             ReLU-61                    [-1, 8]               0
           Linear-62                   [-1, 48]             432
          Dropout-63                   [-1, 48]               0
             ReLU-64                   [-1, 48]               0
          ULTIMUS-65                   [-1, 48]               0
           Linear-66                   [-1, 10]             490
          Dropout-67                   [-1, 10]               0
             ReLU-68                   [-1, 10]               0
================================================================
Total params: 25,978
Trainable params: 25,978
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 3.01
Params size (MB): 0.10
Estimated Total Size (MB): 3.12
----------------------------------------------------------------
```



Training Logs:

```
----------------------------------------------------------------
Model training starts on CIFAR10 dataset
EPOCH: 1
LR: 0.015099999999999999
Loss=2.1314473152160645 Batch_id=97 Accuracy=24.01: 100%|██████████| 98/98 [00:04<00:00, 20.65it/s]
Test set: Average loss: 0.0040, Accuracy: 2721/10000 (27.21%)

validation-accuracy improved from 0 to 27.21, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-1_L1-1_L2-0_val_acc-27.21.h5
EPOCH: 2
LR: 0.04233558282208588
Loss=2.0714080333709717 Batch_id=97 Accuracy=27.50: 100%|██████████| 98/98 [00:04<00:00, 22.77it/s]
Test set: Average loss: 0.0041, Accuracy: 2723/10000 (27.23%)

validation-accuracy improved from 27.21 to 27.23, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-2_L1-1_L2-0_val_acc-27.23.h5
EPOCH: 3
LR: 0.06957116564417178
Loss=2.045386791229248 Batch_id=97 Accuracy=29.00: 100%|██████████| 98/98 [00:04<00:00, 23.45it/s]
Test set: Average loss: 0.0040, Accuracy: 2919/10000 (29.19%)

validation-accuracy improved from 27.23 to 29.19, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-3_L1-1_L2-0_val_acc-29.19.h5
EPOCH: 4
LR: 0.09680674846625767
Loss=2.0425381660461426 Batch_id=97 Accuracy=29.70: 100%|██████████| 98/98 [00:04<00:00, 23.17it/s]
Test set: Average loss: 0.0039, Accuracy: 3043/10000 (30.43%)

validation-accuracy improved from 29.19 to 30.43, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-4_L1-1_L2-0_val_acc-30.43.h5
EPOCH: 5
LR: 0.12404233128834355
Loss=2.1141421794891357 Batch_id=97 Accuracy=30.21: 100%|██████████| 98/98 [00:04<00:00, 22.88it/s]
Test set: Average loss: 0.0039, Accuracy: 2993/10000 (29.93%)

EPOCH: 6
LR: 0.15091971535982815
Loss=2.077439785003662 Batch_id=97 Accuracy=30.23: 100%|██████████| 98/98 [00:04<00:00, 23.27it/s]
Test set: Average loss: 0.0040, Accuracy: 3079/10000 (30.79%)

validation-accuracy improved from 30.43 to 30.79, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-6_L1-1_L2-0_val_acc-30.79.h5
EPOCH: 7
LR: 0.14305182062298602
Loss=2.0244812965393066 Batch_id=97 Accuracy=31.56: 100%|██████████| 98/98 [00:04<00:00, 22.48it/s]
Test set: Average loss: 0.0038, Accuracy: 3409/10000 (34.09%)

validation-accuracy improved from 30.79 to 34.09, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-7_L1-1_L2-0_val_acc-34.09.h5
EPOCH: 8
LR: 0.13518392588614392
Loss=2.0159552097320557 Batch_id=97 Accuracy=32.73: 100%|██████████| 98/98 [00:04<00:00, 23.12it/s]
Test set: Average loss: 0.0038, Accuracy: 3315/10000 (33.15%)

EPOCH: 9
LR: 0.12731603114930182
Loss=1.9737426042556763 Batch_id=97 Accuracy=32.99: 100%|██████████| 98/98 [00:04<00:00, 22.61it/s]
Test set: Average loss: 0.0042, Accuracy: 3046/10000 (30.46%)

EPOCH: 10
LR: 0.11944813641245972
Loss=1.8997420072555542 Batch_id=97 Accuracy=33.85: 100%|██████████| 98/98 [00:04<00:00, 23.47it/s]
Test set: Average loss: 0.0037, Accuracy: 3513/10000 (35.13%)

validation-accuracy improved from 34.09 to 35.13, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-10_L1-1_L2-0_val_acc-35.13.h5
EPOCH: 11
LR: 0.11158024167561761
Loss=1.9875361919403076 Batch_id=97 Accuracy=34.14: 100%|██████████| 98/98 [00:04<00:00, 23.35it/s]
Test set: Average loss: 0.0036, Accuracy: 3741/10000 (37.41%)

validation-accuracy improved from 35.13 to 37.41, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-11_L1-1_L2-0_val_acc-37.41.h5
EPOCH: 12
LR: 0.10371234693877551
Loss=2.019270896911621 Batch_id=97 Accuracy=34.50: 100%|██████████| 98/98 [00:04<00:00, 22.69it/s]
Test set: Average loss: 0.0038, Accuracy: 3487/10000 (34.87%)

EPOCH: 13
LR: 0.09584445220193341
Loss=1.8942344188690186 Batch_id=97 Accuracy=34.74: 100%|██████████| 98/98 [00:04<00:00, 23.35it/s]
Test set: Average loss: 0.0038, Accuracy: 3519/10000 (35.19%)

EPOCH: 14
LR: 0.0879765574650913
Loss=1.8612499237060547 Batch_id=97 Accuracy=35.09: 100%|██████████| 98/98 [00:04<00:00, 22.53it/s]
Test set: Average loss: 0.0036, Accuracy: 3791/10000 (37.91%)

validation-accuracy improved from 37.41 to 37.91, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-14_L1-1_L2-0_val_acc-37.91.h5
EPOCH: 15
LR: 0.0801086627282492
Loss=1.9935224056243896 Batch_id=97 Accuracy=35.34: 100%|██████████| 98/98 [00:04<00:00, 22.95it/s]
Test set: Average loss: 0.0037, Accuracy: 3807/10000 (38.07%)

validation-accuracy improved from 37.91 to 38.07, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-15_L1-1_L2-0_val_acc-38.07.h5
EPOCH: 16
LR: 0.07224076799140709
Loss=1.995123267173767 Batch_id=97 Accuracy=35.61: 100%|██████████| 98/98 [00:04<00:00, 22.64it/s]
Test set: Average loss: 0.0039, Accuracy: 3421/10000 (34.21%)

EPOCH: 17
LR: 0.06437287325456499
Loss=1.9617526531219482 Batch_id=97 Accuracy=35.92: 100%|██████████| 98/98 [00:04<00:00, 23.14it/s]
Test set: Average loss: 0.0035, Accuracy: 3852/10000 (38.52%)

validation-accuracy improved from 38.07 to 38.52, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-17_L1-1_L2-0_val_acc-38.52.h5
EPOCH: 18
LR: 0.05650497851772289
Loss=1.9776537418365479 Batch_id=97 Accuracy=36.37: 100%|██████████| 98/98 [00:04<00:00, 23.19it/s]
Test set: Average loss: 0.0035, Accuracy: 3790/10000 (37.90%)

EPOCH: 19
LR: 0.048637083780880774
Loss=1.8437831401824951 Batch_id=97 Accuracy=36.74: 100%|██████████| 98/98 [00:04<00:00, 22.62it/s]
Test set: Average loss: 0.0036, Accuracy: 3796/10000 (37.96%)

EPOCH: 20
LR: 0.04076918904403867
Loss=1.8770501613616943 Batch_id=97 Accuracy=37.01: 100%|██████████| 98/98 [00:04<00:00, 23.04it/s]
Test set: Average loss: 0.0035, Accuracy: 3873/10000 (38.73%)

validation-accuracy improved from 38.52 to 38.73, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-20_L1-1_L2-0_val_acc-38.73.h5
EPOCH: 21
LR: 0.03290129430719656
Loss=1.8783206939697266 Batch_id=97 Accuracy=37.26: 100%|██████████| 98/98 [00:04<00:00, 22.35it/s]
Test set: Average loss: 0.0036, Accuracy: 3777/10000 (37.77%)

EPOCH: 22
LR: 0.02503339957035447
Loss=1.8417435884475708 Batch_id=97 Accuracy=37.60: 100%|██████████| 98/98 [00:04<00:00, 23.25it/s]
Test set: Average loss: 0.0035, Accuracy: 3994/10000 (39.94%)

validation-accuracy improved from 38.73 to 39.94, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-22_L1-1_L2-0_val_acc-39.94.h5
EPOCH: 23
LR: 0.01716550483351237
Loss=1.893621802330017 Batch_id=97 Accuracy=37.94: 100%|██████████| 98/98 [00:04<00:00, 22.59it/s]
Test set: Average loss: 0.0034, Accuracy: 4012/10000 (40.12%)

validation-accuracy improved from 39.94 to 40.12, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-23_L1-1_L2-0_val_acc-40.12.h5
EPOCH: 24
LR: 0.009297610096670267
Loss=1.8284498453140259 Batch_id=97 Accuracy=38.24: 100%|██████████| 98/98 [00:04<00:00, 22.88it/s]
Test set: Average loss: 0.0034, Accuracy: 4110/10000 (41.10%)

validation-accuracy improved from 40.12 to 41.1, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-24_L1-1_L2-0_val_acc-41.1.h5
```


#### Few details on the Model:

Please refer [EVA8_session9_assignment_model.py](https://github.com/ojhajayant/EVA8_API/blob/main/models/EVA8_session9_assignment_model.py)
All the model files now reside in the same location like this EVA8_session9_assignment_model.py file

```python
!git clone https://git@github.com/ojhajayant//EVA8_API.git
 ```


###  Following shows the training/validation loss plot:

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_9/train_test_loss_plt.png "Logo Title Text 1")

### Misclassified Images:

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_9/misclassified_images.png "Logo Title Text 1")

###  Confusion Matrix & Classification Reports:

For Best saved Model:

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_9/confusion%20matrix.png "Logo Title Text 1")















