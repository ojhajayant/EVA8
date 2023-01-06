Session_2.5:

# Implementation for a neural network that can:
1.   take 2 inputs :
> an image from the MNIST dataset (say 5), and a random number between 0 and 9, (say 7)


2.  and gives two outputs(as in example below) :
> for example: the "number" that was represented by the MNIST image (predict 5), and the 
"sum" of this number with the random number and the input image to the network 
(predict 5 + 7 = 12)


Please refer the [notebook](https://github.com/ojhajayant/EVA8/blob/main/session_2.5/EVA8_session_2_5_final_Jayant_Ojha.ipynb) 

## Data representation & data generation strategy:

### Custom dataset-class required here:
The constructor takes in following params:
 
 1. **train_data (dict):** A dictionary having two columns(/keys):-1. "mnist_input" {ndarry: (60000,28,28)} 
    (representing the numpy-format MNIST training data, and named as "train_mnist_set_array", this is kept 
	in numpy format, so as to be able to use the "transform" callable, provided as argument ("transform_in1")
	for this custom-dataset class, so as to convert to the required Tensor format, with all required posible 
	list of transforms applied to it &  2. "rand_num_input"{Tensor:(60000,10)} (representing the Random-number
	input from the [0,9] in one-hot-encoded form, named as "train_rand_num_set_array_one_hot", this does not 
	require any transform, hence though there is a 2nd transform-interface provided for the 2nd input of this
	overall neural network (provided as argument "transform_in2"), butinterface not used.    
	
 2. **test_data (dict):** A dictionary having two columns(/keys):-1. "mnist_input" {ndarry: (10000,28,28)} 
    (representing the numpy-format MNIST test data, and named as "test_mnist_set_array", this is kept in numpy 
	format, so as to be able to use the "transform" callable, provided as argument ("transform_in1") for this 
	custom-dataset class, so as to convert to the required Tensor format, with all required posible list of 
	transforms applied to it &  2. "rand_num_input"{Tensor:(60000,10)} (representing the Random-number input 
	from the [0,9] in one-hot-encoded form, named as "test_rand_num_set_array_one_hot", this does not require 
	any transform, hence though, there is a 2nd transform-interface provided for the 2nd input of this overall 
	neural network (provided as argument "transform_in2"), but interface not used.
    
 3. **train_target (dict):** A dictionary having two columns(/keys):-1. "mnist_output"{Tensor: (60000,)} 
    (representing the Tensor-format MNIST train-labels/targets, and named as "train_mnist_set_array_targets".
     &  2. "sum_of_mnist_rand_num_output"{Tensor:(60000,19)} (representing the sum output expected to be from the
    [0,18] in one-hot-encoded form, named as "train_sum_of_mnist_rand_num_set_array_one_hot".
    
 4. **test_target (dict):** A dictionary having two columns(/keys):-1. "mnist_output"{Tensor: (10000,)} 
    (representing the Tensor-format MNIST test-labels/targets, and named as "test_mnist_set_array_targets".
     &  2. "sum_of_mnist_rand_num_output"{Tensor:(60000,19)} (representing the sum output expected to be from 
    the [0,18] in one-hot-encoded form, named as "test_sum_of_mnist_rand_num_set_array_one_hot".
  
 5. **train:**: True if the dataset will be used in train mode(by default kept True)
 
 6. **test:**: True if the dataset will be used in test mode(by default kept False).Assert if both train 
    & test flag true simultaneously.
	
 7. **transform_in1:**: a "callable" data-transform interface provided for the part# 1 of the neural network
    (made up of the CNN layers & catering to the mnist image input) By default kept as None.
 
 8. **transform_in2:**: a "callable" data-transform interface provided for the part# 2 of the neural network
    (made up of the FC-layers & catering to the random number input) By default kept as None.
 
## Combining the two inputs (making a multi input/output model)

##### For the neural network(NN), the mnsit image goes thru the given layering below, which, with an input image size of 28x28x1.This forms the first "part" of the NN.Actually here, this layering is leveraged off the earlier session assignment, but now merged to cater to the  additional rand-num input and sum_of_mnist_rand_rum output too.The additional set of inputs and outputs are taken thru a layering of fully connected layers.

Input Channels/Image  |  Conv2d/Transform      | Output Channels | RF
---------------------|--------------|----------------------|----------------------
`28x28x1`              | `(3x3x1)x32`   |      `28x28x32**(padding=1)`  |      `3x3`|      
` `              | `ReLU`   |      ` `  |      ` ` 
**28x28x32**             | **(3x3x32)x64**  |      **28x28x64**(padding=1)** |      **5x5**
** **             | **ReLU**   |     ** **  |     ** **      
*28x28x64*             |   *MP(2x2)*    |      *14x14x64*   |      *10x10*  
**14x14x64**             | **(3x3x64)x128**  |      **14x14x128**(padding=1)** |      **12x12**  
** **             | **ReLU**   |     ** **  |     ** **                                         
*14x14x128*             | *(3x3x128)x256*  |      *14x14x256** (padding=1)*    |      *14x14* 
** **             | *ReLU*   |     * *   |     * *
**14x14x256**             | **MP(2x2)**  |      **7x7x256** |      **28x28** 
**7x7x256**               | **(3x3x256)x512**  |      **5x5x512** (no padding)**  |      **30x30** 
** **             | **ReLU**   |     ** **  |     ** **    
**5x5x512**               | **(3x3x512)x1024**  |      **3x3x1024** (no padding)**  |      **32x32** 
** **             | **ReLU**   |     ** **  |     ** **    
*3x3x1024*               | *(3x3x1024)x10*  |      *1x1x10** (no padding)*    |      *34x34*     

   

>Please note that for the FC-part a provision is made to potentially go for 2 inputs as well (though 
currently only one input rand_num goes thru), In case in future experiment an additional input is required
 that can also be concatenated along (in all one-hot-enoded formats)

The Net class construct hence provides these args:
>num_inputs_for_fc_part=1 (provision for concatenated 2 outputs, currently only one)

>sum_out_features=19 (width 19 one hot encoded sum_out related FC has these out features)

>num_hidden_fc_features=100 (a hidden in between FC layer requires these, by default 100)

## Evaluation of results we finally got:
- Training happened on the GPU (i.e.CUDA Available? True  device:  cuda  )
- final validation/test accuracy---99% (for MNIST image input) & 1% (for the random-num/sum-out combination)
- Logs for 10 epoch runs are shown below:
EPOCH: 1
loss_in1=0.1638360321521759  loss_in2=2.5868406295776367 Accuracy_in1= 52448/60000 (87.41)%Accuracy_in2= 593/60000 (0.99)%batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 20.00it/s]

Test set: Average loss_in1: 0.0592, Accuracy_in1: 9798/10000 (98%)


Test set: Average loss_in2: 2.5501, Accuracy_in2: 104/10000 (1%)

EPOCH: 2
loss_in1=0.02141326665878296  loss_in2=2.411694049835205 Accuracy_in1= 58965/60000 (98.28)%Accuracy_in2= 593/60000 (0.99)%batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 29.17it/s]

Test set: Average loss_in1: 0.0395, Accuracy_in1: 9867/10000 (99%)


Test set: Average loss_in2: 2.3789, Accuracy_in2: 104/10000 (1%)

EPOCH: 3
loss_in1=0.0722644254565239  loss_in2=2.325798511505127 Accuracy_in1= 59299/60000 (98.83)%Accuracy_in2= 593/60000 (0.99)%batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.66it/s]

Test set: Average loss_in1: 0.0335, Accuracy_in1: 9892/10000 (99%)


Test set: Average loss_in2: 2.3362, Accuracy_in2: 104/10000 (1%)

EPOCH: 4
loss_in1=0.03132213279604912  loss_in2=2.325326919555664 Accuracy_in1= 59529/60000 (99.22)%Accuracy_in2= 593/60000 (0.99)%batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.46it/s]

Test set: Average loss_in1: 0.0268, Accuracy_in1: 9901/10000 (99%)


Test set: Average loss_in2: 2.3230, Accuracy_in2: 104/10000 (1%)

EPOCH: 5
loss_in1=0.02715432643890381  loss_in2=2.311291456222534 Accuracy_in1= 59665/60000 (99.44)%Accuracy_in2= 593/60000 (0.99)%batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.31it/s]

Test set: Average loss_in1: 0.0246, Accuracy_in1: 9919/10000 (99%)


Test set: Average loss_in2: 2.3163, Accuracy_in2: 104/10000 (1%)

EPOCH: 6
loss_in1=0.00439819460734725  loss_in2=2.3146703243255615 Accuracy_in1= 59733/60000 (99.56)%Accuracy_in2= 593/60000 (0.99)%batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.19it/s]

Test set: Average loss_in1: 0.0320, Accuracy_in1: 9896/10000 (99%)


Test set: Average loss_in2: 2.3130, Accuracy_in2: 104/10000 (1%)

EPOCH: 7
loss_in1=0.0021874813828617334  loss_in2=2.3381497859954834 Accuracy_in1= 59747/60000 (99.58)%Accuracy_in2= 593/60000 (0.99)%batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.02it/s]

Test set: Average loss_in1: 0.0257, Accuracy_in1: 9916/10000 (99%)


Test set: Average loss_in2: 2.3081, Accuracy_in2: 104/10000 (1%)

EPOCH: 8
loss_in1=0.010930695571005344  loss_in2=2.319537401199341 Accuracy_in1= 59831/60000 (99.72)%Accuracy_in2= 593/60000 (0.99)%batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 27.91it/s]

Test set: Average loss_in1: 0.0251, Accuracy_in1: 9914/10000 (99%)


Test set: Average loss_in2: 2.3129, Accuracy_in2: 104/10000 (1%)

EPOCH: 9
loss_in1=0.0013021514751017094  loss_in2=2.324920654296875 Accuracy_in1= 59833/60000 (99.72)%Accuracy_in2= 593/60000 (0.99)%batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 27.88it/s]

Test set: Average loss_in1: 0.0248, Accuracy_in1: 9929/10000 (99%)


Test set: Average loss_in2: 2.3122, Accuracy_in2: 104/10000 (1%)

EPOCH: 10
loss_in1=0.0033002549316734076  loss_in2=2.316932439804077 Accuracy_in1= 59892/60000 (99.82)%Accuracy_in2= 593/60000 (0.99)%batch_id=468: 100%|██████████| 469/469 [00:17<00:00, 27.57it/s]

Test set: Average loss_in1: 0.0321, Accuracy_in1: 9913/10000 (99%)


Test set: Average loss_in2: 2.3111, Accuracy_in2: 104/10000 (1%)

	- First observation:  the loss_in1(relates to the training for MNIST path) values are decreasing gradually,
	  but loss_in2(relates to the training for rand-in/sum_out path) doesn't come down and same is clear for the 
	  Test-side losses and accuracies.
	- Hence we can conclude that the 2nd part of the network doesn't learn(Accuracy_in2 = ~1%).While the MNIST
	  image path shows proper learning (Accuracy_in1 = ~99+%)

## Loss Function picked and why:
```python
#For the model instantiated as below:
output_in1, output_in2 = model(data_in1, data_in2)
#two separate instances as in this case, it is
#required to havetwo different loss functions
#to evaluate preds of model & backpropagate
loss_in1 = F.cross_entropy(output_in1, target_in1)
loss_in2 = F.cross_entropy(output_in2, target_in2)
:
:
#Infact 2 separate instance of optims will also be required
#to upadte the related parameter groups as below:
#Please note, for the below to apply, appropriate strings
#'conv' & 'fc' are inserted to the related layer namimgs
optimizer_in1 = optim.SGD(params=[param for name, param in
                                  model.named_parameters() if
                                  'conv' in name], lr=0.01, momentum=0.9)
optimizer_in2 = optim.SGD(params=[param for name, param in
                                  model.named_parameters() if
                                  'fc' in name], lr=0.01, momentum=0.9)
 ```
F.cross_entropy as a loss function was used here. It combines the log_softmax 
function with the negative log likelihood loss, a measure of the difference 
between the predicted probability distribution and the true distribution.

