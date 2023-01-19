# Session_4

## Achieve 99.4% (must be consistently shown in last few epochs, and not a one-time achievement) in  less than or equal to 15 Epochs, with less than 10000 Parameters (additional credits for doing this in less than 8000 parameters) and do this in exactly 3 steps!

- total number of parameters---7,684
- @ 3rd attempt validation/test accuracy within 15 epochs---99.48%

#### Given below is the description of 3 attempts, made to achieve the required goals (meeting 99.4%+ accuracy, under the required parameters: <=10K and within 15 epochs) :

### First Attempt:

Please refer the [notebook](https://github.com/ojhajayant/EVA8/blob/main/session_4/EVA8_session4_assignment_1st_attempt.ipynb)

Apart from setting up & stabilizing the infrastructure, for overall training and inference, Priority here in the 1st attempt is fixing up the architecture and underlying elements, which can form the foundation on which to build upon.
Hence, an architecture as shown in the table below is chosen as a "baseline" one.As while solving the earlier session assignment (and also as covered thoroughly in the last "coding-drill-down" session) two architectural components: BatchNorm & DropOut came out to be two important components to be applied on any Conv2d-layer (except the last one), hence for this session assignment, with very limited number of attempts to go thru (i.e. exactly 3 steps), starting out with inclusion of both these components under this 1st "baseline" neural network architecture (as mentioned before, the excercises on effects of gradual inclusion of both components on neural-net performance, was already undertaken while solving the earlier assignment, hence feel fair enough to start out the first attempt architecture, with both of them included under this new "baseline")

Here is the layering, which is being used to achieve the receptive-field and parameter targets.

Input Channels/Image  |  Conv2d/Transform      | Output Channels | RF
---------------------|--------------|----------------------|----------------------
`28x28x1`              | `(3x3x1)x10`   |      `26x26x10`  |      `3x3`  **INPUT BLOCK** [actually just one conv2d 3x3 layer(image-conv2d layer)]    
` `              | `BN(10)`   |      ` `  |      ` `
` `              | `Dropout(2.9%)`   |      ` `  |      ` `
` `              | `ReLU`   |      ` `  |      ` ` 
`26x26x10`             | `(3x3x10)x10`  |      `24x24x10` |      `5x5`     **CONVOLUTION BLOCK 1** [1st conv2d 3x3 layer for this block ]
` `              | `BN(10)`   |      ` `  |      ` `
` `              | `Dropout(2.9%)`   |      ` `  |      ` `
` `              | `ReLU`   |      ` `  |      ` ` 
`24x24x10`             | `(3x3x10)x15`  |      `22x22x15` |      `7x7`     **CONVOLUTION BLOCK 1** [2nd conv2d 3x3 layer for this block ]
` `              | `BN(15)`   |      ` `  |      ` `
` `              | `Dropout(2.9%)`   |      ` `  |      ` `
` `              | `ReLU`   |      ` `  |      ` ` 
**22x22x15**             |   **MP(2x2)**    |     **11x11x15**   |     **8x8**  **TRANSITION BLOCK 1**  [1st maxpool(2,2) layer for this block]                     
**11x11x15**             | **(1x1x15)x10**  |     **11x11x10**   |      **8x8**  **TRANSITION BLOCK 1**  [2nd 1 conv2d 1x1 layer for this block]  
** **             | **BN(10)**   |     ** **  |     ** **                    
** **             | **Dropout(2.9%)**   |     ** **  |     ** **
** **             | **ReLU**   |     ** **  |     ** **      
*11x11x10*             | *(3x3x10)x10*  |      *9x9x10* |      *12x12*  **CONVOLUTION BLOCK 2** [1st conv2d 3x3 layer for this block]
** **            | *BN(10)*   |     * *   |     * * 
** **             | *Dropout(2.9%)*   |     * *   |     * * 
** **             | *ReLU*   |     ** **  |    ** **                       
*9x9x10*             | *(3x3x10)x10*  |      *7x7x10* |      *16x16*  **CONVOLUTION BLOCK 2** [2nd conv2d 3x3 layer for this block] 
** **            | *BN(16)*   |     * *   |     * * 
** **             | *Dropout(2.9%)*   |     * *   |     * * 
** **             | *ReLU*   |     ** **  |    ** **    
*7x7x10*               | *(3x3x10)x32*  |      *5x5x32*  |      *20x20*  **CONVOLUTION BLOCK 2** [3rd conv2d 3x3 layer for this block, this 32 number of output channels at this stage provides a CAPACITY BOOST for the overall network as explained ahead]  
** **            | *BN(32)*   |     * *   |     * * 
** **             | *Dropout(2.9%)*   |     * *   |     * * 
** **             | *ReLU*   |     ** **  |    ** **   
5x5x32               | GAP  LAYER (kernel_size=(5,5)   |      1x1x32          | `28x28` (20 + (5-1)x2 = 28) **OUTPUT BLOCK** [GAP layer]
`1x1x32`               | `(1x1x32)x10`  |      `1x1x10`    |      `28x28` (28 + (1-1)x10 = 28) **OUTPUT BLOCK** [1 conv2d 1x1 layer]
` `             | `log_SoftMax`   |     ` `  |     ` ` 

 above has 4 "components":
 
 1. **INPUT BLOCK:** One initial, "Input Block" at the begining, to convolve over the "original image"  channel(s), due to the opposing constraints of meeting an accuracy number with lesser parameters possible,
    choosing  10 number of kernels for this 'separate' layer (which feeds in to the next "CONVOLUTION BLOCK 1",
    explained below).This 1 initial layer & two following layers (under "CONVOLUTION BLOCK 1") provide receptive
    field of 7x7 pixels(3->5->7) sufficient for the MNIST dataset's edges & gradient generation.No padding used.    
	
 2. **CONVOLUTION BLOCK 1:** placed after the first "Input Block" layer, 2 layers of 3x3 conv2d operators, with 10 & 15 o/p channels at   each layer position respectively.Thus providing 15 channels at the output.No padding used.
    
 3. **TRANSITION BLOCK 1:** max-pool(k=2,s=2) and a-1x1-feature-merger kernel, following the 'CONVOLUTION BLOCK 1'.Provides 10 o/p channels.No padding used.
    
 4. **CONVOLUTION BLOCK 2:** These are 3 layers of 3x3 conv2d operators, with 10, 10, 32 o/p channels at  each layer position respectively.Here point to be noted, is the use of a "capacity boosting" number of o/p channels at the end. (reason explained in the next point below)
  
 5. **OUTPUT BLOCK:**: GAP layer followed by a 1x1 operator (which actually resembles a fully-connected(FC) layer in this case. A noteworthy point (which relates to the "capacity" element of the overall netowrk (NW)) is that the FC (the 1x1 conv2d behaviour here) will work better, in generating the final 10 "class-features" to be used by the log_softmax.If the inputs to it, have more "features points", i.e. for example if we do a 16->10 conversion, vs a 32->10 conversion, we can expect the 10 class-features (for softmax to decide)      generated by a 32->10 conversion will be more "robust" (as compared to say a 16->10 conversion)
     Forunately, with an already "frugal" choice of numbers like 10, 15 etc, made for earlier layer (and also "frugality" in the total number of layers, which still provide an overall RF as 28, i.e. at least just equal to the original image/object size in this case), we can go ahead with choosing 32 here.

Given below is the number of parameters thus obtained, which is well under 8k.
> Total params: 7,684

> Trainable params: 7,684

> Non-trainable params: 0

Also please note that out of all the other tested % ranges for dropout, it was found for this NW that 2.9% dropout worked best.

###### Targets:
	- expect the max Validation Accuracy to reach at least: ~99.1% (i.e. 0.1s in the decimal places, as we are in this case
	  starting out with both BatchNorm + DropOut included in the first baseline architecture)
	- The total params in all these attempts has to be under 10K.
###### Results (1st Attempt):
	- max Validation Accuracy reached: 99.14% 
	- max Train Accuracy reached: 99.17%
	- Total params: 7,684
  
![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_4/curves_attempt1.png "Logo Title Text 1")
##### Analysis:
	- Actually, for the given Network capacity & limited number of epochs,  with both BatchNorm & Dropout included, we are
	  going good here.There isn't much overfitting, but without added image augmentation and/or learning rate tunings we 
	  can't hit the target on accuracy.Actually, under a limited scope of 15 epochs and comparing with what some addition 
	  of proper image augmenation and/or LR-tuning can do to the validation accuracy (in relation to the training accuracy),
	  we can still call this a case of "overfiting".
	- But, we would now on, in next attempts like to see, how with added Image augmentation the NW-performance gets affected.

##### File Link:  [notebook](https://github.com/ojhajayant/EVA8/blob/main/session_4/EVA8_session4_assignment_1st_attempt.ipynb) 


### Second Attempt:

Please refer the [notebook](https://github.com/ojhajayant/EVA8/blob/main/session_4/EVA8_session4_assignment_2nd_attempt.ipynb)

In this 2nd attempt too, the network structure /architecture hasn't changed.Under this 2nd attempt, a torchvision transform is applied on the images, that randomly rotates an image by a random angle within the range of (-7 degrees, +7 degrees), with the choice made here. The rotation is applied randomly to the image on the specified degree range. For example, if degrees=7 as is the case here, the image will be rotated by a random angle between -7 and 7 degrees.Hence, an architecture as shown in the table below is still chosen as a "baseline" one.



###### Targets:
	- expect the max Validation Accuracy to reach at least: ~99.2% (i.e. 0.2s in the decimal places)
	- The total params in all these attempts has to be under 10K.
###### Results (2nd Attempt):
	- max Validation Accuracy reached: 99.33% 
	- max Train Accuracy reached: 98.95%
	- Total params are still : 7,684 (as even with the 2nd attempt, same underlying structure is taken forward)
  
![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_4/curves_attempt2.png "Logo Title Text 1")
##### Analysis:
	- Under this 2nd attempt, a torchvision transform is applied on the images, that randomly rotates an image by a random
	  angle within the range of (-7 degrees, +7 degrees), with the choice made here. The rotation is applied randomly to 
	  the image on the specified degree range. For example, if degrees=7 as is the case here, the image will be rotated by 
	  a random angle between -7 and 7 degrees. We find that as compared to the first attempt, we have overcome the overfitting 
	  a lot and reached test accuracies up to 99.33% while train accuracy still under 99%.
	- Hence a scope further opens up for us to try out LR-tuning on top of the same in the next (last) attempt to push things
	  further ahead.

##### File Link:  [notebook](https://github.com/ojhajayant/EVA8/blob/main/session_4/EVA8_session4_assignment_2nd_attempt.ipynb) 


### Third Attempt:

In this 2nd attempt too, the network structure /architecture hasn't changed. During one experiment on the learning rate sweeps, one learning rate region below 0.07 'flashed' a better accuracy number, hence using just the StepLR way with step size 1, with gamma such that every epoch the reduction in learning rate would be 0.1/TOTAL_EPOCHS:
```
(i.e. 
with init_learning_rate = 0.07
gamma x init_learning_rate = (init_learning_rate - (0.1/TOTAL_EPOCHS)   while step_size = 1)
```
Guided by such equation on gamma the LR swept thru ranges as: LR: [0.07, 0.06333333333333334, 0.057301587301587305, 0.0518442932728647.....]
In this range during last few epochs, and once in betwen, it flashed these 99.4+% accuracies.

Please refer the [notebook](https://github.com/ojhajayant/EVA8/blob/main/session_4/EVA8_session4_assignment_3rd_attempt.ipynb)


###### Targets:
	- expect the max Validation Accuracy to reach the target: ~99.4% 
	- The total params in all these attempts has to be under 10K.
###### Results (3rd Attempt):
	- max Validation Accuracy reached: 99.48% 
	- max Train Accuracy reached: 99.18%
	- Total params are still : 7,684 (as even with the 3rd attempt, same underlying structure is taken forward)
  
![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_4/curves_attempt3.png "Logo Title Text 1")
##### Analysis:
	- During experiments on learning rate sweeps, one learning rate region below 0.07 'flashed' a better accuracy number,
	  hence using just the StepLR way with step size 1, with gamma such that every epoch the reduction in learning rate 
	  would be 0.1/TOTAL_EPOCHS:

```
(i.e. 
with init_learning_rate = 0.07
gamma x init_learning_rate = (init_learning_rate - (0.1/TOTAL_EPOCHS)   while step_size = 1)
```
	- Guided by such equation on gamma the LR swept thru ranges as: LR: [0.07, 0.06333333333333334, 0.057301587301587305,
	  0.0518442932728647.....] In this range during last few epochs, and once in betwen, it flashed these 99.4+% 
	  accuracies.

##### File Link:  [notebook](https://github.com/ojhajayant/eva/blob/master/week5/S5_assignment_3rd_attempt.ipynb) 

