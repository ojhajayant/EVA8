Session_3:

PART 1:

# For a Neural Network, with weights initialized as shown in the diagram below, write an excel-sheet showing the calculation for BackPropagation:
![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_3/Neural_Network_diagram.png "Logo Title Text 1")
# Also explain each Major step with required equations.Give snapshots for the loss curve under a list of learning rates: [0.1, 0.2, 0.5, 0.8, 1.0, 2.0] 

Please refer the [xls sheet](https://github.com/ojhajayant/EVA8/blob/main/session_2.5/EVA8_session_2_5_final_Jayant_Ojha.ipynb) 

#### Major Step 1:

This step actually is about calculations related to a "forward propagation", given an intiailized set of weights( $w1=0.15$, $w2=0.2$, $w3=0.25$, $w4=0.3$, $w5=0.4$, $w6=0.45$, $w7=0.5$ & $w8=0.55$), a given set of inputs( $i1=0.05$, $i2=0.1$) along with a given set of target/ground-truth values( $t1=0.01$, $t2=0.99$)
The equations are as follows:

$$ h_1 = w_1 i_1 + w_2 i_2 $$	

$$ h_2 = w_3 i_1 + w_4 i_2 $$	

$$ a__h_1 = σ(h_1) = 1 \over (1 + exp(-h_1)) $$

$$ a__h_2 = σ(h_2) = 1 \over (1 + exp(-h_2)) $$		

$$ o_1 = w_5 a__h_1 + w_6 a__h_2 $$

$$ o_2 = w_7 a__h_1 + w_8 a__h_2 $$		

$$ a__o_1 = σ(o_1) = 1 \over (1 + exp(-o_1)) $$		

$$ a__o_2 = σ(o_2) = 1 \over (1 + exp(-o_2)) $$		

$$ E__Total = E_1 + E_2 $$	

$$ E_1 = ½ * (t_1 - a__o_1)² $$	

$$ E_2 = ½ * (t_2 - a__o_2)² $$		



![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_3/loss_curve_lr_0.1.png "Logo Title Text 1")
![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_3/loss_curve_lr_0.2.png "Logo Title Text 1")
![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_3/loss_curve_lr_0.5.png "Logo Title Text 1")
![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_3/loss_curve_lr_0.8.png "Logo Title Text 1")
![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_3/loss_curve_lr_1.0.png "Logo Title Text 1")
![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_3/loss_curve_lr_2.0.png "Logo Title Text 1")


## xyz:

### ucw:
Tfggh:
 

##### cvg


PART 2:
