Session_3:

PART 1:

# For a Neural Network, with weights initialized as shown in the diagram below, write an excel-sheet showing the calculation for BackPropagation:
![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_3/Neural_Network_diagram.png "Logo Title Text 1")
# Also explain each Major step with required equations.Give snapshots for the loss curve under a list of learning rates: [0.1, 0.2, 0.5, 0.8, 1.0, 2.0] 

Please refer the [xls sheet](https://github.com/ojhajayant/EVA8/blob/main/session_2.5/EVA8_session_2_5_final_Jayant_Ojha.ipynb) 

#### Major Step 1:

This step actually is about calculations related to a "forward propagation", given an intiailized set of weights( $w1=0.15$, $w2=0.2$, $w3=0.25$, $w4=0.3$, $w5=0.4$, $w6=0.45$, $w7=0.5$ & $w8=0.55$), a given set of inputs( $i1=0.05$, $i2=0.1$) along with a given set of target/ground-truth values( $t1=0.01$, $t2=0.99$)
The equations are as follows:

h1 = w1*i1 + w2*i2

h2 = w3*i1 + w4*i2

a_h1 = σ(h1) = 1/(1 + e^{-h1})

a_h2 = σ(h2) = 1/(1 + e^{-h2})

o1=w5*a_h1 + w6*a_h2

o2=w7*a_h1 + w8*a_h2

a_o1=σ(o1) = 1/(1 + e^{-o1})

a_o2=σ(o2) = 1/(1 + e^{-o2})

E_Total=E1+E2

E1=½ * (t1 - a_o1)²

E2 = ½ * (t2 - a_o2)²





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
