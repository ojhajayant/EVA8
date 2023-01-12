# Session_3:

# PART 1:

## For a Neural Network, with weights initialized as shown in the diagram below, write an excel-sheet showing the calculation for BackPropagation:
![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_3/Neural_Network_diagram.png "Logo Title Text 1")
## Also explain each Major step with required equations.Give snapshots for the loss curve under a list of learning rates: [0.1, 0.2, 0.5, 0.8, 1.0, 2.0] 

Please refer the [xls sheet](https://github.com/ojhajayant/EVA8/blob/main/session_2.5/EVA8_session_2_5_final_Jayant_Ojha.ipynb) which has calculations based on the 6 "Major-Steps" as described below.Multiple sheets within this workbook, represent calculations for the "convergence" of the neural network, under different learning rates (esp. related to the list:  [0.1, 0.2, 0.5, 0.8, 1.0, 2.0] but under similar intial values of weights, same set of input and target values too.)

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

Under this next step, while back propagating the E_Total behind the hidden layer, the next set of parameters/weights we want to update are: $w1$, $w2$, $w3$ & $w4$.Our purpose through this major-step is to be able to calculate the partial derivative of E_Total w.r.t. $w1$, $w2$, $w3$ & $w4$.Under this step we represent each of ∂E_total/∂w1 (and others w.r.t. $w2$, $w3$ & $w4$) as set of 4 chain rule equations where first represents: ∂E_total/∂w1 equal to (∂E_total/∂a_h1)*(∂a_h1/∂h1)*(∂h1/∂w1) and same extends to other 3.



$$ ∂E_total / ∂w1 = (∂E_total / ∂a_{h_1} ) * (∂a_{h_1} / ∂h1 ) * (∂h1 / ∂w1 )	$$
				
$$ ∂E_total / ∂w2 = (∂E_total / ∂a_{h_1} ) * (∂a_{h_1} / ∂h1 ) * (∂h1 / ∂w2 )	$$	
			
$$ ∂E_total / ∂w3 = (∂E_total / ∂a_{h_2} ) * (∂a_{h_2} / ∂h2 ) * (∂h2 / ∂w3 )	 $$
				
$$ ∂E_total / ∂w4 = (∂E_total / ∂a_{h_2} ) * (∂a_{h_2} / ∂h2 ) * (∂h2 / ∂w4 )	 $$	
			
	
#### Major Step 6:
Under this final step we expand the equations obtained under step-4 to get the final set of equations for the partial derivative of E_Total w.r.t. $w1$, $w2$, $w3$ & $w4$ as given below:

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

	- observation:  the convergence improves with the increase in the learning rates (at least, for the given set of values)


# PART 2:
