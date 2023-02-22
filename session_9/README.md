
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

	> cmd : train
	
	> IPYNB_ENV : True
	
	> use_albumentations : True
	
	> SEED : 1
	
	> dataset : CIFAR10
	
	> img_size : (32, 32)
	
	> batch_size : 512
	
	> epochs : 24
	
	> criterion : CrossEntropyLoss()
	
	> init_lr : 0.0001
	
	> end_lr : 0.1
	
	> max_lr_epochs : 5
	
	> lr_range_test_epochs : 150
	
	> best_lr : 0.868
	
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
	
	> best_model :  


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
Model training starts on CIFAR10 dataset
EPOCH: 1
LR: 0.0868
Loss=2.4302804470062256 Batch_id=97 Accuracy=9.99: 100%|██████████| 98/98 [00:08<00:00, 11.62it/s]

Test set: Average loss: 0.0046, Accuracy: 1000/10000 (10.00%)

validation-accuracy improved from 0 to 10.0, saving model to /content/EVA8_API/./saved_models/CIFAR10_model_epoch-1_L1-1_L2-0_val_acc-10.0.h5
EPOCH: 2
LR: 0.24335950920245397
Loss=2.391749858856201 Batch_id=97 Accuracy=9.98: 100%|██████████| 98/98 [00:07<00:00, 13.56it/s]

Test set: Average loss: 0.0046, Accuracy: 1000/10000 (10.00%)

EPOCH: 3
LR: 0.39991901840490796
Loss=2.3557114601135254 Batch_id=97 Accuracy=9.99: 100%|██████████| 98/98 [00:07<00:00, 13.77it/s]

Test set: Average loss: 0.0046, Accuracy: 1000/10000 (10.00%)

EPOCH: 4
LR: 0.5564785276073619
Loss=3.1600539684295654 Batch_id=97 Accuracy=10.02: 100%|██████████| 98/98 [00:07<00:00, 12.78it/s]

Test set: Average loss: 0.0046, Accuracy: 1000/10000 (10.00%)

EPOCH: 5
LR: 0.7130380368098159
Loss=3.044361114501953 Batch_id=97 Accuracy=10.00: 100%|██████████| 98/98 [00:07<00:00, 13.53it/s]

Test set: Average loss: 0.0046, Accuracy: 1000/10000 (10.00%)

EPOCH: 6
LR: 0.8675384962406015
Loss=8.653585433959961 Batch_id=97 Accuracy=10.01: 100%|██████████| 98/98 [00:07<00:00, 13.35it/s]

Test set: Average loss: 0.0046, Accuracy: 1000/10000 (10.00%)

EPOCH: 7
LR: 0.8223111278195488
Loss=8.557609558105469 Batch_id=97 Accuracy=10.00: 100%|██████████| 98/98 [00:07<00:00, 12.79it/s]

Test set: Average loss: 0.0046, Accuracy: 1000/10000 (10.00%)

EPOCH: 8
LR: 0.7770837593984963
Loss=8.4616060256958 Batch_id=97 Accuracy=10.00: 100%|██████████| 98/98 [00:07<00:00, 13.63it/s]

Test set: Average loss: 0.0046, Accuracy: 1000/10000 (10.00%)

EPOCH: 9
LR: 0.7318563909774436
Loss=8.404401779174805 Batch_id=97 Accuracy=10.00: 100%|██████████| 98/98 [00:07<00:00, 13.36it/s]

Test set: Average loss: 0.0046, Accuracy: 1000/10000 (10.00%)

EPOCH: 10
LR: 0.6866290225563909
Loss=8.345022201538086 Batch_id=97 Accuracy=10.00: 100%|██████████| 98/98 [00:07<00:00, 12.70it/s]

Test set: Average loss: 0.0046, Accuracy: 1000/10000 (10.00%)

EPOCH: 11
LR: 0.6414016541353383
Loss=9.504398345947266 Batch_id=97 Accuracy=10.00: 100%|██████████| 98/98 [00:07<00:00, 13.16it/s]

Test set: Average loss: 0.0046, Accuracy: 1000/10000 (10.00%)

EPOCH: 12
LR: 0.5961742857142858
Loss=9.258179664611816 Batch_id=97 Accuracy=10.00: 100%|██████████| 98/98 [00:07<00:00, 13.06it/s]

Test set: Average loss: 0.0046, Accuracy: 1000/10000 (10.00%)

EPOCH: 13
LR: 0.5509469172932331
Loss=9.140510559082031 Batch_id=97 Accuracy=10.00: 100%|██████████| 98/98 [00:07<00:00, 12.47it/s]

Test set: Average loss: 0.0046, Accuracy: 1000/10000 (10.00%)

EPOCH: 14
LR: 0.5057195488721804
Loss=9.061992645263672 Batch_id=97 Accuracy=10.00: 100%|██████████| 98/98 [00:07<00:00, 13.30it/s]

Test set: Average loss: 0.0046, Accuracy: 1000/10000 (10.00%)

EPOCH: 15
LR: 0.46049218045112783
Loss=8.999622344970703 Batch_id=97 Accuracy=10.00: 100%|██████████| 98/98 [00:07<00:00, 13.17it/s]

Test set: Average loss: 0.0046, Accuracy: 1000/10000 (10.00%)

EPOCH: 16
LR: 0.4152648120300752
Loss=9.352771759033203 Batch_id=97 Accuracy=10.00: 100%|██████████| 98/98 [00:07<00:00, 12.62it/s]

Test set: Average loss: 0.0046, Accuracy: 1000/10000 (10.00%)

EPOCH: 17
LR: 0.37003744360902263
Loss=9.290525436401367 Batch_id=97 Accuracy=10.00: 100%|██████████| 98/98 [00:07<00:00, 13.24it/s]

Test set: Average loss: 0.0046, Accuracy: 1000/10000 (10.00%)

EPOCH: 18
LR: 0.3248100751879699
Loss=9.244192123413086 Batch_id=97 Accuracy=10.00: 100%|██████████| 98/98 [00:07<00:00, 13.31it/s]

Test set: Average loss: 0.0046, Accuracy: 1000/10000 (10.00%)

EPOCH: 19
LR: 0.2795827067669172
Loss=9.206872940063477 Batch_id=97 Accuracy=10.00: 100%|██████████| 98/98 [00:07<00:00, 12.56it/s]

Test set: Average loss: 0.0046, Accuracy: 1000/10000 (10.00%)

EPOCH: 20
LR: 0.23435533834586464
Loss=9.177281379699707 Batch_id=97 Accuracy=10.00: 100%|██████████| 98/98 [00:07<00:00, 13.19it/s]

Test set: Average loss: 0.0046, Accuracy: 1000/10000 (10.00%)

EPOCH: 21
LR: 0.18912796992481196
Loss=9.154308319091797 Batch_id=97 Accuracy=10.00: 100%|██████████| 98/98 [00:07<00:00, 13.30it/s]

Test set: Average loss: 0.0046, Accuracy: 1000/10000 (10.00%)

EPOCH: 22
LR: 0.14390060150375938
Loss=9.137107849121094 Batch_id=97 Accuracy=10.00: 100%|██████████| 98/98 [00:07<00:00, 12.57it/s]

Test set: Average loss: 0.0046, Accuracy: 1000/10000 (10.00%)

EPOCH: 23
LR: 0.0986732330827067
Loss=9.125844955444336 Batch_id=97 Accuracy=10.00: 100%|██████████| 98/98 [00:07<00:00, 12.84it/s]

Test set: Average loss: 0.0046, Accuracy: 1000/10000 (10.00%)

EPOCH: 24
LR: 0.05344586466165413
Loss=9.120853424072266 Batch_id=97 Accuracy=10.00: 100%|██████████| 98/98 [00:07<00:00, 13.34it/s]

Test set: Average loss: 0.0046, Accuracy: 1000/10000 (10.00%)
```


#### Few details on the Model:

Please refer [EVA8_session9_assignment_model.py](https://github.com/ojhajayant/EVA8_API/blob/main/models/EVA8_session9_assignment_model.py)
All the model files now reside in the same location like this EVA8_session9_assignment_model.py file

```python
!git clone https://git@github.com/ojhajayant//EVA8_API.git
 ```


###  Following shows the training/validation loss plot:

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_9/train_test_loss_plt.png "Logo Title Text 1")


###  Confusion Matrix & Classification Reports:

For Best saved Model:

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_9/confusion%20matrix.png "Logo Title Text 1")















