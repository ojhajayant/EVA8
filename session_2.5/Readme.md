# Session_2.5:

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

## Evaluation of results:
- Training happened on the GPU (i.e.CUDA Available? True  device:  cuda  )
- final validation/test accuracy---99% (for MNIST image input) & 1% (for the random-num/sum-out combination)
- Logs for 10 epoch runs are shown below:
EPOCH: 1
loss_in1=0.16420362889766693  loss_in2=2.5868406295776367 Accuracy_in1= 52410/60000 (87.35)%Accuracy_in2= 593/60000 (0.99)%batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 29.27it/s]

Test set: Average loss_in1: 0.0587, Accuracy_in1: 9806/10000 (98%)


Test set: Average loss_in2: 2.5501, Accuracy_in2: 104/10000 (1%)

EPOCH: 2
loss_in1=0.025337709113955498  loss_in2=2.411694049835205 Accuracy_in1= 58975/60000 (98.29)%Accuracy_in2= 593/60000 (0.99)%batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.41it/s]

Test set: Average loss_in1: 0.0374, Accuracy_in1: 9879/10000 (99%)


Test set: Average loss_in2: 2.3789, Accuracy_in2: 104/10000 (1%)

EPOCH: 3
loss_in1=0.0661570355296135  loss_in2=2.325798511505127 Accuracy_in1= 59303/60000 (98.84)%Accuracy_in2= 593/60000 (0.99)%batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.74it/s]

Test set: Average loss_in1: 0.0331, Accuracy_in1: 9894/10000 (99%)


Test set: Average loss_in2: 2.3362, Accuracy_in2: 104/10000 (1%)

EPOCH: 4
loss_in1=0.03862898051738739  loss_in2=2.325326919555664 Accuracy_in1= 59537/60000 (99.23)%Accuracy_in2= 593/60000 (0.99)%batch_id=468: 100%|██████████| 469/469 [00:17<00:00, 26.82it/s]

Test set: Average loss_in1: 0.0266, Accuracy_in1: 9912/10000 (99%)


Test set: Average loss_in2: 2.3230, Accuracy_in2: 104/10000 (1%)

EPOCH: 5
loss_in1=0.018267376348376274  loss_in2=2.311291456222534 Accuracy_in1= 59669/60000 (99.45)%Accuracy_in2= 593/60000 (0.99)%batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.22it/s]

Test set: Average loss_in1: 0.0250, Accuracy_in1: 9919/10000 (99%)


Test set: Average loss_in2: 2.3163, Accuracy_in2: 104/10000 (1%)

EPOCH: 6
loss_in1=0.004356232937425375  loss_in2=2.3146703243255615 Accuracy_in1= 59730/60000 (99.55)%Accuracy_in2= 593/60000 (0.99)%batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.12it/s]

Test set: Average loss_in1: 0.0308, Accuracy_in1: 9894/10000 (99%)


Test set: Average loss_in2: 2.3130, Accuracy_in2: 104/10000 (1%)

EPOCH: 7
loss_in1=0.002201579511165619  loss_in2=2.3381497859954834 Accuracy_in1= 59766/60000 (99.61)%Accuracy_in2= 593/60000 (0.99)%batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 27.72it/s]

Test set: Average loss_in1: 0.0267, Accuracy_in1: 9921/10000 (99%)


Test set: Average loss_in2: 2.3081, Accuracy_in2: 104/10000 (1%)

EPOCH: 8
loss_in1=0.00935713667422533  loss_in2=2.319537401199341 Accuracy_in1= 59832/60000 (99.72)%Accuracy_in2= 593/60000 (0.99)%batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 27.73it/s]

Test set: Average loss_in1: 0.0269, Accuracy_in1: 9919/10000 (99%)


Test set: Average loss_in2: 2.3129, Accuracy_in2: 104/10000 (1%)

EPOCH: 9
loss_in1=0.0032400053460150957  loss_in2=2.324920654296875 Accuracy_in1= 59829/60000 (99.72)%Accuracy_in2= 593/60000 (0.99)%batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 27.62it/s]

Test set: Average loss_in1: 0.0255, Accuracy_in1: 9930/10000 (99%)


Test set: Average loss_in2: 2.3122, Accuracy_in2: 104/10000 (1%)

EPOCH: 10
loss_in1=0.01408284530043602  loss_in2=2.316932439804077 Accuracy_in1= 59893/60000 (99.82)%Accuracy_in2= 593/60000 (0.99)%batch_id=468: 100%|██████████| 469/469 [00:17<00:00, 27.48it/s]

Test set: Average loss_in1: 0.0321, Accuracy_in1: 9910/10000 (99%)


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






### With a recent "deadline-extension", opportunistically adding here, an [implementation](https://github.com/ojhajayant/EVA8/blob/main/session_2.5/EVA8_session_2_5_with_additional_mnist_label_input_final_Jayant_Ojha.ipynb) which apart from the mnist-image & rand-num inputs takes in current-mnist-label as an additional input (i.e. the 2nd Fully-connected (FC) part has 2 inputs rand-num & mnist-label now)

```python
#it now instantiates the network as below
model = Net(2).to(device) # The "2nd Part"(the fully-connected(FC)-part)
                          # has 2 inputs rand-in + curent-mnist-label.
						  
						  :
						  :
#while under train_model & test_model functions the following 
#change takes place (everything else remaining same as the 
#earlier implementation)
output_in1, output_in2 = model(data_in1, #mnist image
                               data_in2, #rand-num
                               # & current-mnist-label
                               one_hot_custom(target_in1,
                                              width=10,
					      device=device))
 ```
### Under this implementation as well:

- Training happened on the GPU (i.e.CUDA Available? True  device:  cuda  )
- final validation/test accuracy---99% (for MNIST image input) & 1% (for the random-num+current-mnist-label/sum-out combination)
- Logs for 10 epoch runs are shown below:

EPOCH: 1
loss_in1=0.06289205700159073  loss_in2=2.1441330909729004 Accuracy_in1= 51535/60000 (85.89)%Accuracy_in2= 593/60000 (0.99)%batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 20.31it/s]

Test set: Average loss_in1: 0.0620, Accuracy_in1: 9792/10000 (98%)


Test set: Average loss_in2: 2.1526, Accuracy_in2: 104/10000 (1%)

EPOCH: 2
loss_in1=0.03607624024152756  loss_in2=0.7177146673202515 Accuracy_in1= 58951/60000 (98.25)%Accuracy_in2= 593/60000 (0.99)%batch_id=468: 100%|██████████| 469/469 [00:17<00:00, 27.41it/s]

Test set: Average loss_in1: 0.0342, Accuracy_in1: 9890/10000 (99%)


Test set: Average loss_in2: 0.7073, Accuracy_in2: 104/10000 (1%)

EPOCH: 3
loss_in1=0.023446200415492058  loss_in2=0.11855971813201904 Accuracy_in1= 59277/60000 (98.80)%Accuracy_in2= 593/60000 (0.99)%batch_id=468: 100%|██████████| 469/469 [00:17<00:00, 27.53it/s]

Test set: Average loss_in1: 0.0289, Accuracy_in1: 9902/10000 (99%)


Test set: Average loss_in2: 0.1149, Accuracy_in2: 104/10000 (1%)

EPOCH: 4
loss_in1=0.024031415581703186  loss_in2=0.034561820328235626 Accuracy_in1= 59488/60000 (99.15)%Accuracy_in2= 593/60000 (0.99)%batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 27.73it/s]

Test set: Average loss_in1: 0.0254, Accuracy_in1: 9917/10000 (99%)


Test set: Average loss_in2: 0.0361, Accuracy_in2: 104/10000 (1%)

EPOCH: 5
loss_in1=0.005684105679392815  loss_in2=0.019983984529972076 Accuracy_in1= 59595/60000 (99.33)%Accuracy_in2= 593/60000 (0.99)%batch_id=468: 100%|██████████| 469/469 [00:17<00:00, 26.65it/s]

Test set: Average loss_in1: 0.0246, Accuracy_in1: 9912/10000 (99%)


Test set: Average loss_in2: 0.0183, Accuracy_in2: 104/10000 (1%)

EPOCH: 6
loss_in1=0.018205055966973305  loss_in2=0.011255479417741299 Accuracy_in1= 59718/60000 (99.53)%Accuracy_in2= 593/60000 (0.99)%batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 27.65it/s]

Test set: Average loss_in1: 0.0286, Accuracy_in1: 9907/10000 (99%)


Test set: Average loss_in2: 0.0116, Accuracy_in2: 104/10000 (1%)

EPOCH: 7
loss_in1=0.0009696869528852403  loss_in2=0.0076314411126077175 Accuracy_in1= 59741/60000 (99.57)%Accuracy_in2= 593/60000 (0.99)%batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 27.60it/s]

Test set: Average loss_in1: 0.0251, Accuracy_in1: 9921/10000 (99%)


Test set: Average loss_in2: 0.0082, Accuracy_in2: 104/10000 (1%)

EPOCH: 8
loss_in1=7.546281995018944e-05  loss_in2=0.0060789394192397594 Accuracy_in1= 59830/60000 (99.72)%Accuracy_in2= 593/60000 (0.99)%batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 27.64it/s]

Test set: Average loss_in1: 0.0248, Accuracy_in1: 9922/10000 (99%)


Test set: Average loss_in2: 0.0063, Accuracy_in2: 104/10000 (1%)

EPOCH: 9
loss_in1=0.00016558513743802905  loss_in2=0.004717394709587097 Accuracy_in1= 59841/60000 (99.73)%Accuracy_in2= 593/60000 (0.99)%batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 27.73it/s]

Test set: Average loss_in1: 0.0253, Accuracy_in1: 9919/10000 (99%)


Test set: Average loss_in2: 0.0050, Accuracy_in2: 104/10000 (1%)

EPOCH: 10
loss_in1=0.0038362108170986176  loss_in2=0.004112758673727512 Accuracy_in1= 59874/60000 (99.79)%Accuracy_in2= 593/60000 (0.99)%batch_id=468: 100%|██████████| 469/469 [00:17<00:00, 27.55it/s]

Test set: Average loss_in1: 0.0243, Accuracy_in1: 9932/10000 (99%)


Test set: Average loss_in2: 0.0041, Accuracy_in2: 104/10000 (1%)
