# Session_3:

# PART 1:

## For a Neural Network, with weights initialized as shown in the diagram below, write an excel-sheet showing the calculation for BackPropagation:
![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_3/Neural_Network_diagram.png "Logo Title Text 1")
## Also explain each Major step with required equations.Give snapshots for the loss curve under a list of learning rates: [0.1, 0.2, 0.5, 0.8, 1.0, 2.0] 

Please refer the [xls sheet](https://github.com/ojhajayant/EVA8/blob/main/session_3/BackPropogationCalculation.xlsx) which has calculations based on the 6 "Major-Steps" as described below.Multiple sheets within this workbook, represent calculations for the "convergence" of the neural network, under different learning rates (esp. related to the list:  [0.1, 0.2, 0.5, 0.8, 1.0, 2.0] but under similar intial values of weights, same set of input and target values too.)
One example screenshot from one of the sheets of the xls sheet is as shown here (for learning rate 1.0)
![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_3/Back_Prop_calc_lr_1.0.png "Logo Title Text 1")

#### Major Step 1:

This step actually is about calculations related to a "forward propagation", given an intiailized set of weights( $w1=0.15$, $w2=0.2$, $w3=0.25$, $w4=0.3$, $w5=0.4$, $w6=0.45$, $w7=0.5$ & $w8=0.55$), a given set of inputs( $i1=0.05$, $i2=0.1$) along with a given set of target/ground-truth values( $t1=0.01$, $t2=0.99$)
From the neural network diagram above, the equations are as follows:

$$ h1 = w1 * i1 + w2 * i2 $$

$$ h2 = w3 * i1 + w4 * i2 $$

$$ a_{h_1} = σ(h1) = 1/(1 + e^{-h1}) $$

$$ a_{h_2} = σ(h2) = 1/(1 + e^{-h2}) $$

$$ o1 = w5 * a_{h_1} + w6 * a_{h_2} $$

$$ o2 = w7 * a_{h_1} + w8 * a_{h_2} $$

$$ a_{o_1} = σ(o1) = 1/(1 + e^{-o1}) $$

$$ a_{o_2} = σ(o2) = 1/(1 + e^{-o2}) $$

$$ E_Total = E1 + E2 $$

$$ E1 = ½  *  (t1 - a_{o_1})² $$

$$ E2 = ½  *  (t2 - a_{o_2})² $$

Here the L2 loss function is used where for example t1 is the target or ground truth and a_o1 is the NN-prediction.Here E_Total represents sum of E1 & E2, as we set out to train the network while reducing the overall loss.

#### Major Step 2:

Under this next step, while back propagating the E_Total, the first set of parameters/weights we want to update are: $w5$, $w6$, $w7$ & $w8$.Our purpose through this major-step is to be able to calculate the partial derivative of E_Total w.r.t. $w5$, $w6$, $w7$ & $w8$.Hence, under this step we demonstrate how to calculate just one out of these i.e. the partial derivative of E_Total w.r.t. $w5$, so that in the next major step # 3, we just extend the same formation, to calculate this partial derivative for E_Total w.r.t. the remaining other last-level weights(i.e. $w6$, $w7$ & $w8$).

The mathematical equations for $w5$ are as follows:

$$ ∂E_total / ∂w5 = ∂(E1 + E2) / ∂w5 $$
					
$$ ∂E_total / ∂w5 = ∂E1 / ∂w5 $$
					
$$ ∂E_total / ∂w5 = ∂E1 / ∂w5 = (∂E1 / ∂a_{o_1}) * (∂a_{o_1} / ∂o1) * (∂o1 / ∂w5) $$
					
$$ ∂E1 / ∂a_{o_1}  = ∂(½ * (t1 - a_{o_1})²) / ∂a_{o_1} = a_{o_1} - t1 $$
					
$$ ∂a_{o_1}/∂o1 = ∂(σ(o1))/∂o1 = a_{o_1} * (1 - a_{o_1} ) $$
					
$$ ∂o1 / ∂w5 = a_{h_1} $$

Please note in the 2nd equation above, $∂E_total / ∂w5$ became $∂E1 / ∂w5$ as for $w5$ $E2$ wasn't involved.And 3rd equation above represents our intution in mathematical terms (i.e. chain-rule) & finally th last 3 equations in this step represent the 3 multiplicands under the chain of derivatives.
					

#### Major Step 3:

Under this next step, we calculate the final value for the derivative of E_Total w.r.t. $w5$ and extend the same to $w6$ to $w8$

$$ ∂E_total / ∂w5 = (a_{o_1} - t1) * a_{o_1} * (1 - a_{o_1} ) * a_{h_1} $$ 	

$$ ∂E_total / ∂w6 = (a_{o_1} - t1) * a_{o_1} * (1 - a_{o_1} ) * a_{h_2}	$$

$$ ∂E_total / ∂w7 = (a_{o_2} - t2) * a_{o_2} * (1 - a_{o_2} ) * a_{h_1}	$$	

$$ ∂E_total / ∂w8 = (a_{o_2} - t2) * a_{o_2} * (1 - a_{o_2} ) * a_{h_1}	$$


#### Major Step 4:

Under this next step, we step back to the activation outputs for the hidden layer & calculate the value for the derivative of E_Total w.r.t. $a_{h_1}$ and extend the same to $a_{h_2}$.While doing the ∂ETotal/∂a_h1, we realize that both E1 and E2 paths are involved under this calculation, hence both ∂E1/∂a_h1  & ∂E2/∂a_h1 need to be calculated first.While looking at ∂E1/∂a_h1, we realize that intutively this can be calculated using the chain-rule: (∂E1/∂a_o1) * (∂a_o1/∂o1) * (∂o1/∂a_h1). The 2 initial equations below represent the same rule where the property of derivative of a sigmoid being equal to sigmoid*(1-sigmoid) is also applied.While under 3rd equation the initial first 2 equations are summed up, thus calculating the final value of ∂E_total/∂a_h1. Last equation under this part represents a similar extension to calculate ∂E_total/∂a_h2.

$$ ∂E1 / ∂a_{h_1} = (a_{o_1} - t1) * a_{o_1} * (1 - a_{o_1} ) * w5 $$		
					
$$ ∂E2 / ∂a_{h_1} = (a_{o_2} - t2) * a_{o_2} * (1 - a_{o_2} ) * w7 $$	
							
$$ ∂E_total / ∂a_{h_1} = (a_{o_1} - t1 ) * a_{o_1} * (1 - a_{o_1} ) * w5  + (a_{o_2} - t2 ) * a_{o_2} * (1 - a_{o_2} ) * w7 $$
								
$$ ∂E_total / ∂a_{h_2} = (a_{o_1} - t1 ) * a_{o_1} * (1 - a_{o_1} ) * w6 + (a_{o_2} - t2) * a_{o_2} * (1 - a_{o_2} ) * w8 $$
								
 
				
#### Major Step 5:

Under this next step, while back propagating the E_Total behind the hidden layer, the next set of parameters/weights we want to update are: $w1$, $w2$, $w3$ & $w4$.Our purpose through this major-step is to be able to calculate the partial derivative of E_Total w.r.t. $w1$, $w2$, $w3$ & $w4$.Under this step we represent each of ∂E_total/∂w1 (and others w.r.t. $w2$, $w3$ & $w4$) as set of 4 chain rule equations where first represents: ∂E_total/∂w1 equal to (∂E_total/∂a_h1) * (∂a_h1/∂h1) * (∂h1/∂w1) and same extends to other 3.



$$ ∂E_total / ∂w1 = (∂E_total / ∂a_{h_1} ) * (∂a_{h_1} / ∂h1 ) * (∂h1 / ∂w1 )	$$
				
$$ ∂E_total / ∂w2 = (∂E_total / ∂a_{h_1} ) * (∂a_{h_1} / ∂h1 ) * (∂h1 / ∂w2 )	$$	
			
$$ ∂E_total / ∂w3 = (∂E_total / ∂a_{h_2} ) * (∂a_{h_2} / ∂h2 ) * (∂h2 / ∂w3 )	 $$
				
$$ ∂E_total / ∂w4 = (∂E_total / ∂a_{h_2} ) * (∂a_{h_2} / ∂h2 ) * (∂h2 / ∂w4 )	 $$	
			
	
#### Major Step 6:
Under this final step we expand the equations obtained under step-5 to get the final set of equations for the partial derivative of E_Total w.r.t. $w1$, $w2$, $w3$ & $w4$ as given below:

$$ ∂E_total / ∂w1 = ((a_{o_1} - t1) *  a_{o_1} * (1 - a_{o_1}) * w5 + (a_{o_2} - t2) * a_{o_2} * (1 - a_{o_2}) * w7) * a_{h_1} * (1 - a_{h_1}) * i1 $$
										
$$ ∂E_total / ∂w2 = ((a_{o_1} - t1) * a_{o_1} * (1 - a_{o_1}) * w5 + (a_{o_2} - t2) * a_{o_2} * (1 - a_{o_2}) * w7) * a_{h_1} * (1 - a_{h_1}) * i2	$$
									
$$ ∂E_total / ∂w3 = ((a_{o_1} - t1) * a_{o_1} * (1 - a_{o_1}) * w6 + (a_{o_2} - t2) * a_{o_2} * (1 - a_{o_2}) * w8) * a_{h_2} * (1 - a_{h_2}) * i1 $$
										
$$ ∂E_total / ∂w4 = ((a_{o_1} - t1) * a_{o_1} * (1 - a_{o_1}) * w6 + (a_{o_2} - t2) * a_{o_2} * (1 - a_{o_2}) * w8) * a_{h_2} * (1 - a_{h_2}) * i2	$$									

#### Loss Curve with learning rate ɳ = 0.1 : 

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_3/loss_curve_lr_0.1.png "Logo Title Text 1")

#### Loss Curve with learning rate ɳ = 0.2 : 

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_3/loss_curve_lr_0.2.png "Logo Title Text 1")

#### Loss Curve with learning rate ɳ = 0.5 : 

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_3/loss_curve_lr_0.5.png "Logo Title Text 1")

#### Loss Curve with learning rate ɳ = 0.8 : 

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_3/loss_curve_lr_0.8.png "Logo Title Text 1")

#### Loss Curve with learning rate ɳ = 1.0 : 

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_3/loss_curve_lr_1.0.png "Logo Title Text 1")

#### Loss Curve with learning rate ɳ = 2.0 : 

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_3/loss_curve_lr_2.0.png "Logo Title Text 1")

	- observation:  the convergence improves with the increase in the learning rates (at least, for the provided set of lr values here)


# PART 2:

## For MNIST Dataset, meet 99.4%+ accuracy, but under the required parameters:i.e. <=20K and within 20 epochs)

- total number of parameters---14,112
- final validation/test accuracy---99.42%

Please refer the [notebook](https://github.com/ojhajayant/EVA8/blob/main/session_3/EVA8_session_3_assignment_Jayant_Ojha.ipynb) 

## Logs:
epoch=1 loss=0.187565416097641 batch_id=937: 100%|██████████| 938/938 [00:22<00:00, 42.53it/s]
Epoch 1--Train set: Average loss: 0.1876, Training Accuracy: 50284/60000 (83.81%)



Epoch 1--Test set: Average loss: 0.0839, Validation Accuracy: 9832/10000 (98.32%)

epoch=2 loss=0.12101959437131882 batch_id=937: 100%|██████████| 938/938 [00:22<00:00, 41.00it/s]
Epoch 2--Train set: Average loss: 0.1210, Training Accuracy: 57775/60000 (96.29%)



Epoch 2--Test set: Average loss: 0.0470, Validation Accuracy: 9871/10000 (98.71%)

epoch=3 loss=0.03076682984828949 batch_id=937: 100%|██████████| 938/938 [00:23<00:00, 39.64it/s]
Epoch 3--Train set: Average loss: 0.0308, Training Accuracy: 58309/60000 (97.18%)



Epoch 3--Test set: Average loss: 0.0399, Validation Accuracy: 9889/10000 (98.89%)

epoch=4 loss=0.282584011554718 batch_id=937: 100%|██████████| 938/938 [00:23<00:00, 40.55it/s]
Epoch 4--Train set: Average loss: 0.2826, Training Accuracy: 58587/60000 (97.64%)



Epoch 4--Test set: Average loss: 0.0390, Validation Accuracy: 9879/10000 (98.79%)

epoch=5 loss=0.020755881443619728 batch_id=937: 100%|██████████| 938/938 [00:22<00:00, 41.76it/s]
Epoch 5--Train set: Average loss: 0.0208, Training Accuracy: 58740/60000 (97.90%)



Epoch 5--Test set: Average loss: 0.0326, Validation Accuracy: 9907/10000 (99.07%)

epoch=6 loss=0.06976652890443802 batch_id=937: 100%|██████████| 938/938 [00:22<00:00, 41.77it/s]
Epoch 6--Train set: Average loss: 0.0698, Training Accuracy: 58850/60000 (98.08%)



Epoch 6--Test set: Average loss: 0.0274, Validation Accuracy: 9920/10000 (99.20%)

epoch=7 loss=0.08803491294384003 batch_id=937: 100%|██████████| 938/938 [00:22<00:00, 41.69it/s]

Epoch 7--Train set: Average loss: 0.0880, Training Accuracy: 58882/60000 (98.14%)


Epoch 7--Test set: Average loss: 0.0265, Validation Accuracy: 9926/10000 (99.26%)

epoch=8 loss=0.0430348739027977 batch_id=937: 100%|██████████| 938/938 [00:22<00:00, 41.55it/s]

Epoch 8--Train set: Average loss: 0.0430, Training Accuracy: 58953/60000 (98.25%)


Epoch 8--Test set: Average loss: 0.0238, Validation Accuracy: 9932/10000 (99.32%)

epoch=9 loss=0.031756848096847534 batch_id=937: 100%|██████████| 938/938 [00:22<00:00, 41.58it/s]
Epoch 9--Train set: Average loss: 0.0318, Training Accuracy: 58997/60000 (98.33%)



Epoch 9--Test set: Average loss: 0.0232, Validation Accuracy: 9929/10000 (99.29%)

epoch=10 loss=0.09574046730995178 batch_id=937: 100%|██████████| 938/938 [00:22<00:00, 41.72it/s]
Epoch 10--Train set: Average loss: 0.0957, Training Accuracy: 59032/60000 (98.39%)



Epoch 10--Test set: Average loss: 0.0240, Validation Accuracy: 9928/10000 (99.28%)

epoch=11 loss=0.17473085224628448 batch_id=937: 100%|██████████| 938/938 [00:22<00:00, 41.86it/s]
Epoch 11--Train set: Average loss: 0.1747, Training Accuracy: 59113/60000 (98.52%)



Epoch 11--Test set: Average loss: 0.0228, Validation Accuracy: 9930/10000 (99.30%)

epoch=12 loss=0.07037123292684555 batch_id=937: 100%|██████████| 938/938 [00:22<00:00, 41.58it/s]
Epoch 12--Train set: Average loss: 0.0704, Training Accuracy: 59151/60000 (98.58%)



Epoch 12--Test set: Average loss: 0.0225, Validation Accuracy: 9923/10000 (99.23%)

epoch=13 loss=0.039098143577575684 batch_id=937: 100%|██████████| 938/938 [00:22<00:00, 41.04it/s]
Epoch 13--Train set: Average loss: 0.0391, Training Accuracy: 59137/60000 (98.56%)



Epoch 13--Test set: Average loss: 0.0220, Validation Accuracy: 9936/10000 (99.36%)

epoch=14 loss=0.032910604029893875 batch_id=937: 100%|██████████| 938/938 [00:22<00:00, 41.39it/s]
Epoch 14--Train set: Average loss: 0.0329, Training Accuracy: 59177/60000 (98.63%)



Epoch 14--Test set: Average loss: 0.0227, Validation Accuracy: 9931/10000 (99.31%)

epoch=15 loss=0.007501773536205292 batch_id=937: 100%|██████████| 938/938 [00:22<00:00, 41.25it/s]
Epoch 15--Train set: Average loss: 0.0075, Training Accuracy: 59190/60000 (98.65%)



Epoch 15--Test set: Average loss: 0.0214, Validation Accuracy: 9932/10000 (99.32%)

epoch=16 loss=0.06674277782440186 batch_id=937: 100%|██████████| 938/938 [00:22<00:00, 41.57it/s]
Epoch 16--Train set: Average loss: 0.0667, Training Accuracy: 59238/60000 (98.73%)



Epoch 16--Test set: Average loss: 0.0211, Validation Accuracy: 9936/10000 (99.36%)

epoch=17 loss=0.00403211172670126 batch_id=937: 100%|██████████| 938/938 [00:22<00:00, 41.20it/s]
Epoch 17--Train set: Average loss: 0.0040, Training Accuracy: 59226/60000 (98.71%)



Epoch 17--Test set: Average loss: 0.0207, Validation Accuracy: 9932/10000 (99.32%)

epoch=18 loss=0.08990251272916794 batch_id=937: 100%|██████████| 938/938 [00:22<00:00, 41.25it/s]
Epoch 18--Train set: Average loss: 0.0899, Training Accuracy: 59221/60000 (98.70%)



Epoch 18--Test set: Average loss: 0.0208, Validation Accuracy: 9930/10000 (99.30%)

epoch=19 loss=0.012829605489969254 batch_id=937: 100%|██████████| 938/938 [00:22<00:00, 41.48it/s]
Epoch 19--Train set: Average loss: 0.0128, Training Accuracy: 59267/60000 (98.78%)



Epoch 19--Test set: Average loss: 0.0201, Validation Accuracy: 9941/10000 (99.41%)

epoch=20 loss=0.11119721829891205 batch_id=937: 100%|██████████| 938/938 [00:22<00:00, 41.77it/s]
Epoch 20--Train set: Average loss: 0.1112, Training Accuracy: 59255/60000 (98.76%)



Epoch 20--Test set: Average loss: 0.0179, Validation Accuracy: 9942/10000 (99.42%)

#### Following dscribes a step-by-step approach, taken to achieve the required goals (i.e. meeting 99.4%+ accuracy, but under the required parameters:i.e. <=20K and within 20 epochs) :
 
First of all, "fix up" the required architecture/layer-organization...in terms of choosing the number of kernels for each layer, followed a 'multiple-of-8' numbers (from 8, 16, 24, 32...etc):

 #### Architecture has 3 "components":
 
 1. One, separate initial,"image-conv2d" layer at the begining, to convolve over the "original image" channels, am initially providing 8 number of kernels for this 'separate' layer (which feeds in to the next  "block" of "2-conv2d-layers").Very important to note here that, 1 initial layer + The 2 following layers(of "2-conv2d-layer" "block") collectively provides a receptive field of 7x7 pixels(3->5->7) sufficient for extracting the MNIST datset's edges & gradient.This one is an "evolution"-experiment, where kernel numbers started out as 8 initially, but in the final architecture(which met the requirements) it became 16.
    
	
 2. conv2d-BLOCK with 2 layers (in this case):  This block will be placed after the first "image-conv2d" layer, and one more instance of this block, will    also follow the transition-block (explained below) later. In this evolution-experiment, kernel numbers initially started out as (8-16) for the 'first-2-layer-block' & (8-16) for the 'second-2-layer-block', but in the final architecture(which met the requirements) it became (16-16) for the 'first-2-layer-block' & (24-24) for the 'second-2-layer-block'.
    
    
 3. Transition Blocks: 1st transition layer, made up of 1. max-pool(k=2,s=2) and 2. a-1x1-feature-merger kernel, following the 'first-2-layer-block' &     2nd transition layer, towards the end (following the 2nd conv2d-block) which does NOT have the maxpool (i.e. just has one 1x1-feature-merger operator), and followed by the Global Average Pooling (GAP) layer leading upto the Softmax layer. 


#### Architecture (i.e. in terms of channels used across individual layers):

    
    i.   "image-conv2d" layer: o/p initially 8 channels (becomes 16 in the final one)
    
    ii.  2 similar conv2d blocks, with:
    
              1st layer: (8-16) o/p channels (becomes (16-16) in the final one)
			 
              2nd layer: (8-16) o/p channels (becomes (24-24) in the final one)
	      
    iii. 1x1 conv2d for 2nd transition-layer: 10 o/p channels(for num-classes=10 digits)
    
    
 
Input Channels/Image  |  Conv2d/Transform      | Output Channels | RF
---------------------|--------------|----------------------|----------------------
`28x28x1`              | `(3x3x1)x8`   |      `26x26x8`  |      `3x3`|      
` `              | `ReLU`   |      ` `  |      ` ` 
**26x26x8**             | **(3x3x8)x8**  |      **24x24x8** |      **5x5**
** **             | **ReLU**   |     ** **  |     ** **      
**24x24x8**             | **(3x3x8)x16**  |      **22x22x16** |      **7x7**  
** **             | **ReLU**   |     ** **  |     ** **                       
*22x22x16*             |   *MP(2x2)*    |      *11x11x16*   |      *14x14*                      
*11x11x16*             | *(1x1x16)x8*  |      *11x11x8*    |      *14x14* 
** **             | *ReLU*   |     * *   |     * *
**11x11x8**             | **(3x3x8)x8**  |      **9x9x8** |      **16x16** 
** **             | **ReLU**   |     ** **  |     ** **   
**9x9x8**               | **(3x3x8)x16**  |      **7x7x16**  |      **18x18** 
** **             | **ReLU**   |     ** **  |     ** **    
*7x7x16*               | *(1x1x16)x10*  |      *7x7x10*    |      *18x18*  (NO RELU at the o/p of this layer)    
7x7x10               | GAP  LAYER   |      1x10          |



       
    iv. At the end, following Architecture (same as the first-table above, 
    but with increased number of channels, like below is found to achieve 
    the required goal: 14,112 params, >99.4% accuracy, in less than 
    20 epochs
        

Input Channels/Image  |  Conv2d/Transform      | Output Channels | RF
---------------------|--------------|----------------------|----------------------
`28x28x1`              | `(3x3x1)x16`   |      `26x26x16`  |      `3x3`
` `              | `BN(16)`   |      ` `  |      ` `
` `              | `Dropout(3%)`   |      ` `  |      ` `
` `              | `ReLU`   |      ` `  |      ` `  
**26x26x16**             | **(3x3x16)x16**  |      **24x24x16** |      **5x5** 
** **             | **BN(16)**   |     ** **  |     ** **
** **             | **Dropout(3%)**   |     ** **  |     ** **
** **             | **ReLU**   |     ** **  |     ** **              
**24x24x16**             | **(3x3x16)x16**  |      **22x22x16** |      **7x7** 
** **             | **BN(16)**   |     ** **  |     ** **
** **             | **Dropout(3%)**   |     ** **  |     ** **
** **             | **ReLU**   |     ** **  |     ** **              
*22x22x16*             |   *MP(2x2)*    |      *11x11x16*   |      *14x14*                      
*11x11x16*             | *(1x1x16)x16*  |      *11x11x16*    |      *14x14*   (Here  a conversion from 16 input channels to -> 16 output channels, using 1x1 might look like an 'oddity', but as 1x1 is a feature-merging operator, which works along-with BackProp, to get "contextually-linked" channels as outputs for the upstream layers (from the 'scattered' lower level features)
** **            | *BN(16)*   |     * *   |     * * 
** **             | *Dropout(3%)*   |     * *   |     * * 
** **             | *ReLU*   |     * *   |     * *                         
**11x11x16**             | **(3x3x16)x24**  |      **9x9x24** |      **16x16**  
** **             | **BN(24)**   |     ** **  |     ** **
** **             | **Dropout(3%)**   |     ** **  |     ** **
** **             | **ReLU**   |     ** **  |     ** **                         
**9x9x24**               | **(3x3x24)x24**  |      **7x7x24**  |      **18x18**   
** **             | **BN(24)**   |     ** **  |     ** **
** **             | **Dropout(3%)**   |     ** **  |     ** **
** **             | **ReLU**   |     ** **  |     ** **                          
*7x7x24*               | *(1x1x24)x10*  |      *7x7x10*    |      *18x18*   (NO RELU at the o/p of this layer)   
7x7x10               | GAP  LAYER   |      1x10          |     


#### First NW:

Input Channels/Image  |  Conv2d/Transform      | Output Channels | RF
---------------------|--------------|----------------------|----------------------
`28x28x1`              | `(3x3x1)x8`   |      `26x26x8`  |      `3x3`|      
` `              | `ReLU`   |      ` `  |      ` ` 
**26x26x8**             | **(3x3x8)x8**  |      **24x24x8** |      **5x5**
** **             | **ReLU**   |     ** **  |     ** **      
**24x24x8**             | **(3x3x8)x16**  |      **22x22x16** |      **7x7**  
** **             | **ReLU**   |     ** **  |     ** **                       
*22x22x16*             |   *MP(2x2)*    |      *11x11x16*   |      *14x14*                      
*11x11x16*             | *(1x1x16)x8*  |      *11x11x8*    |      *14x14* 
** **             | *ReLU*   |     * *   |     * *
**11x11x8**             | **(3x3x8)x8**  |      **9x9x8** |      **16x16** 
** **             | **ReLU**   |     ** **  |     ** **   
**9x9x8**               | **(3x3x8)x16**  |      **7x7x16**  |      **18x18** 
** **             | **ReLU**   |     ** **  |     ** **    
*7x7x16*               | *(1x1x16)x10*  |      *7x7x10*    |      *18x18*  (NO RELU at the o/p of this layer)    
7x7x10               | GAP  LAYER   |      1x10          |

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_3/accuracy_01.png "Logo Title Text 1")
![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_3/loss_01.png "Logo Title Text 1")

	- First observation: with total 3,816 parameters, the max validation accuracy reached: ~98.16%
	- The logs and accuracy plot show that there is some overfitting in terms of the accuracy plot, 
	  that the train and test accuracies seem to be running in close step/embrace. With not much 
	  "potential" for the test/validation accuracy to increase, with a corresponding increase in 
	  training accuracy. 
	- Additionally, with required 20 epochs, the "network capacity" seems to be NOT sufficient to 
	  meet the goal. 
	- But, despite an urgent need for "capacity-boost" thru increased channels, we would STILL 
	  like to see, if with an "incremental" addition of Batch-Norm, Dropout, what extra 
	  effects/benefits or disadvantages could be seen?


#### 2nd NW: 

added just the Batch Normalization at each layer(we should expect some increase in accuracy in this case)

Input Channels/Image  |  Conv2d/Transform      | Output Channels | RF
---------------------|--------------|----------------------|----------------------
`28x28x1`              | `(3x3x1)x8`   |      `26x26x8`  |      `3x3`|  
` `              | `BN(8)`   |      ` `  |      ` `
` `              | `ReLU`   |      ` `  |      ` ` 
**26x26x8**             | **(3x3x8)x8**  |      **24x24x8** |      **5x5**
** **             | **BN(8)**   |     ** **  |     ** **
** **             | **ReLU**   |     ** **  |     ** **      
**24x24x8**             | **(3x3x8)x16**  |      **22x22x16** |      **7x7** 
** **             | **BN(16)**   |     ** **  |     ** **
** **             | **ReLU**   |     ** **  |     ** **                       
*22x22x16*             |   *MP(2x2)*    |      *11x11x16*   |      *14x14*                      
*11x11x16*             | *(1x1x16)x8*  |      *11x11x8*    |      *14x14* 
** **            | *BN(8)*   |     * *   |     * * 
** **             | *ReLU*   |     * *   |     * *
**11x11x8**             | **(3x3x8)x8**  |      **9x9x8** |      **16x16** 
** **             | **BN(8)**   |     ** **  |     ** **
** **             | **ReLU**   |     ** **  |     ** **   
**9x9x8**               | **(3x3x8)x16**  |      **7x7x16**  |      **18x18** 
** **             | **BN(16)**   |     ** **  |     ** **
** **             | **ReLU**   |     ** **  |     ** **    
*7x7x16*               | *(1x1x16)x10*  |      *7x7x10*    |      *18x18*  (NO RELU at the o/p of this layer)    
7x7x10               | GAP  LAYER   |      1x10          |

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_3/accuracy_02.png "Logo Title Text 1")
![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_3/loss_02.png "Logo Title Text 1")


	- First observation:  Total params: 3,944 (which is expectedly, higher as compared to the 1st NW)
	  & the max validation accuracy reaches: ~98.95%
	- we got little better in terms of the reached accuracy but expectedly, far from the required 
	  accuracy goal.
	- Also in terms of the accuracy plot & logs, we could see that the further potential for increase
	  for the validation accuracy has not yet opened up (it is still overfitting...i.e. the training
	  accuracy has reached 99.0% but the validation acuracy still under ~98.95%, hence not much scope 
	  for further increase, with a corresponding increase in training accuracy )
	- just like the earlier NW, this one can't meet the goal within the required 20 epochs(capacity
	  boost definitely required)

#### 3rd NW:

added Dropout as well, at each layer, apart from the BN(we should expect lesser overfitting  i.e an increase in the "potential" to increase validation accuracy with a corresponding increase in training-accuracy) i.e. a gap opening up (it was viewed, that a value of 3% for Dropout was giving a better results as compared to other test-ranges from 1-10%)

Input Channels/Image  |  Conv2d/Transform      | Output Channels | RF
---------------------|--------------|----------------------|----------------------
`28x28x1`              | `(3x3x1)x8`   |      `26x26x8`  |      `3x3`|  
` `              | `BN(8)`   |      ` `  |      ` `
` `              | `Dropout(3%)`   |      ` `  |      ` `
` `              | `ReLU`   |      ` `  |      ` ` 
**26x26x8**             | **(3x3x8)x8**  |      **24x24x8** |      **5x5**
** **             | **BN(8)**   |     ** **  |     ** **
** **             | **Dropout(3%)**   |     ** **  |     ** **
** **             | **ReLU**   |     ** **  |     ** **      
**24x24x8**             | **(3x3x8)x16**  |      **22x22x16** |      **7x7** 
** **             | **BN(16)**   |     ** **  |     ** **
** **             | **Dropout(3%)**   |     ** **  |     ** **
** **             | **ReLU**   |     ** **  |     ** **                       
*22x22x16*             |   *MP(2x2)*    |      *11x11x16*   |      *14x14*                      
*11x11x16*             | *(1x1x16)x8*  |      *11x11x8*    |      *14x14* 
** **            | *BN(8)*   |     * *   |     * * 
** **             | *Dropout(3%)*   |     * *   |     * * 
** **             | *ReLU*   |     * *   |     * *
**11x11x8**             | **(3x3x8)x8**  |      **9x9x8** |      **16x16** 
** **             | **BN(8)**   |     ** **  |     ** **
** **             | **Dropout(3%)**   |     ** **  |     ** **
** **             | **ReLU**   |     ** **  |     ** **   
**9x9x8**               | **(3x3x8)x16**  |      **7x7x16**  |      **18x18** 
** **             | **BN(16)**   |     ** **  |     ** **
** **             | **Dropout(3%)**   |     ** **  |     ** **
** **             | **ReLU**   |     ** **  |     ** **    
*7x7x16*               | *(1x1x16)x10*  |      *7x7x10*    |      *18x18*  (NO RELU at the o/p of this layer)    
7x7x10               | GAP  LAYER   |      1x10          |

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_3/accuracy_03.png "Logo Title Text 1")
![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_3/loss_03.png "Logo Title Text 1")


	- First observation:  Total params: 3,944 (which is expectedly,same as compared to the last NW as
	  dropout doesn't add params) & the max validation accuracy reaches: ~98.39% (though this seems 
	  almost comparable to the achieved max in last iteration, but in case of adding Dropout, we have
	  considerably overcome the overfitting issue, by opening up the gap between training and test 
	  accuracies...i.e. now we can see that the test accuracy value like ~98% is achieved while the 
	  training accuracy is still at ~96%, which has opened up the potential for further increase in 
	  the validation accuracy, given we have a required number of epochs and NW-capacity with us.
	- Also in terms of the accuracy plot, both the train and test accuracies seem to be maintaining 
	  almost consistent gap (training accuracy growth is looking stagnant though, but some scope is
	  open)
	- As a final step for this architecture-option we are required to the given Learning rate value 
	  hence that aspect won't be explored further.
	- But to achieve the required goal we will have to go for a capacity increase now, the next 
	  iteration tries to do the same.

#### 4th NW:

while retaining the same template for the Batch Normalization, Dropout and layering, we now increase the channels to follow the layering as given ahead(it was viewed, that a value of 3% for Dropout was giving a better results as compared to other test-ranges from 1-10%) "|16|--|16-16|--|Transition|--|24-24|--|Transition|--|GAP|--|softmax-classifier|" as in the table below

Input Channels/Image  |  Conv2d/Transform      | Output Channels | RF
---------------------|--------------|----------------------|----------------------
`28x28x1`              | `(3x3x1)x16`   |      `26x26x16`  |      `3x3`
` `              | `BN(16)`   |      ` `  |      ` `
` `              | `Dropout(3%)`   |      ` `  |      ` `
` `              | `ReLU`   |      ` `  |      ` `  
**26x26x16**             | **(3x3x16)x16**  |      **24x24x16** |      **5x5** 
** **             | **BN(16)**   |     ** **  |     ** **
** **             | **Dropout(3%)**   |     ** **  |     ** **
** **             | **ReLU**   |     ** **  |     ** **              
**24x24x16**             | **(3x3x16)x16**  |      **22x22x16** |      **7x7** 
** **             | **BN(16)**   |     ** **  |     ** **
** **             | **Dropout(3%)**   |     ** **  |     ** **
** **             | **ReLU**   |     ** **  |     ** **              
*22x22x16*             |   *MP(2x2)*    |      *11x11x16*   |      *14x14*                      
*11x11x16*             | *(1x1x16)x16*  |      *11x11x16*    |      *14x14* (Here  a conversion from 16 input channels to -> 16 output channels, using 1x1 might look like an 'oddity', but as 1x1 is a feature-merging operator, which works along-with BackProp, to get "contextually-linked" channels as outputs for the upstream layers (from the 'scattered' lower level features)  
** **            | *BN(16)*   |     * *   |     * * 
** **             | *Dropout(3%)*   |     * *   |     * * 
** **             | *ReLU*   |     * *   |     * *                         
**11x11x16**             | **(3x3x16)x24**  |      **9x9x24** |      **16x16**  
** **             | **BN(24)**   |     ** **  |     ** **
** **             | **Dropout(3%)**   |     ** **  |     ** **
** **             | **ReLU**   |     ** **  |     ** **                         
**9x9x24**               | **(3x3x24)x24**  |      **7x7x24**  |      **18x18**   
** **             | **BN(24)**   |     ** **  |     ** **
** **             | **Dropout(3%)**   |     ** **  |     ** **
** **             | **ReLU**   |     ** **  |     ** **                          
*7x7x24*               | *(1x1x24)x10*  |      *7x7x10*    |      *18x18*   (NO RELU at the o/p of this layer)   
7x7x10               | GAP  LAYER   |      1x10          |     

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_3/accuracy_04.png "Logo Title Text 1")
![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_3/loss_04.png "Logo Title Text 1")


	- First observation:  Total params: 14,112 (which is expected,given higher NW capacity) & the max
	  validation accuracy reaches: ~99.42%
	- This case meets the required goal of getting an accuracy of 99.4% (appeared 2 times during 
	  training-epochs), the parameters: 14,112 < 20K (requirement) and came under 20 epochs.

