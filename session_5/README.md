# Session_5

### Make 3 versions of 4th Session assignment's best model :
> 1. Network with Group Normalization

> 2. Network with Layer Normalization

> 3. Network with L1 + BN


> Write a single model.py file that includes GN/LN/BN and takes an argument to decide which normalization to include.

> Write a single notebook file to run all the 3 models above for 20 epochs each

> Create these graphs:
> 1. Graph 1: Test/Validation Loss for all 3 models together
> 2. Graph 2: Test/Validation Accuracy for 3 models together

> graphs must have proper annotation

> Find 10 misclassified images for each of the 3 models, and show them as a 5x2 image matrix in 3 separately annotated images.  Achieve 99.4% (must be consistently shown in last few epochs, and not a one-time achievement) in  less than or equal to 15 Epochs, with less than 10000 Parameters (additional credits for doing this in less than 8000 parameters) and do this in exactly 3 steps!


Please refer the [notebook](https://github.com/ojhajayant/EVA8/blob/main/session_5/EVA8_session5_assignment.ipynb) for this assignment solution.

#### Few details on the Model with batch/group/layer norm options:

Please refer [model.py](https://github.com/ojhajayant/EVA8/blob/main/session_5/modular/models/model.py)

First of all, for importing the single model file named "model.py" under which model class named "EVA8_session4_assignment_model" is defined, with capabilities to have either of Batch/Layer/Group normalization, we have to put it as a package-element at the github (so that we can do a pip-install as below and can import it as a package element later on, in the colab notebook as below)

```python
!pip install git+https://git@github.com/ojhajayant/EVA8.git --upgrade 

Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Collecting git+https://****@github.com/ojhajayant/EVA8.git
  Cloning https://****@github.com/ojhajayant/EVA8.git to /tmp/pip-req-build-_vv0i96a
  Running command git clone --filter=blob:none --quiet 'https://****@github.com/ojhajayant/EVA8.git' /tmp/pip-req-build-_vv0i96a
  Resolved https://****@github.com/ojhajayant/EVA8.git to commit a3a82086f39e93297bfbee2bd32774126ad203a3
  Preparing metadata (setup.py) ... done
Building wheels for collected packages: eva8
  Building wheel for eva8 (setup.py) ... done
  Created wheel for eva8: filename=eva8-0.0.0-py3-none-any.whl size=3221 sha256=13c89c20108e24af6e90c535b6c3a81bc9f746f67fa5d314ee2342a78ffe7bb7
  Stored in directory: /tmp/pip-ephem-wheel-cache-15pwa9gt/wheels/91/2b/81/6528ca90c705fbe7f126306baa4c34cd2cbf0ea8fcb5a3dd90
Successfully built eva8
Installing collected packages: eva8
Successfully installed eva8-0.0.0
 ```


Here is one sample conv2d layer implementation to optionally select either of the layers: 	  
```python

class EVA8_session4_assignment_model(nn.Module):
    def __init__(self, normalization='batch'):
        super(EVA8_session4_assignment_model, self).__init__()
        self.normalization = normalization

        # Input Block
        # For nn.Sequential construct to work, it needs a "list" of layers,
        # hence start out with the conv2d layer, as the first element of such
        # "list" and keep appending Norm/Dropout/Relu extra layers to it.
        input_block_conv_layer_0 = [nn.Conv2d(in_channels=1, out_channels=10,
                                              kernel_size=(3, 3),
                                              padding=0, bias=False)]
        if self.normalization == 'batch':
            input_block_conv_layer_0.append(nn.BatchNorm2d(10))
        elif self.normalization == 'layer':
            input_block_conv_layer_0.append(nn.LayerNorm([10, 26, 26]))
        elif self.normalization == 'group':
            input_block_conv_layer_0.append(nn.GroupNorm(5, 10))

        input_block_conv_layer_0.append(nn.Dropout(dropout_value))
        input_block_conv_layer_0.append(nn.ReLU())
        self.convblock1 = nn.Sequential(*input_block_conv_layer_0)  # input_size
        # = 28 output_size = 26 receptive_field = 3

 ```
 
Just to repeat, as also described in session 4 (but with slight modifications under the BN/LN/GN aspect), here is the layering, which is being used to achieve the receptive-field and parameter targets.

Input Channels/Image  |  Conv2d/Transform      | Output Channels | RF
---------------------|--------------|----------------------|----------------------
`28x28x1`              | `(3x3x1)x10`   |      `26x26x10`  |      `3x3`  **INPUT BLOCK** [actually just one conv2d 3x3 layer(image-conv2d layer)]    
` `              | `BN(10) or LN(10, 26, 26) or GN(5, 10)`   |      ` `  |      ` `
` `              | `Dropout(2.9%)`   |      ` `  |      ` `
` `              | `ReLU`   |      ` `  |      ` ` 
`26x26x10`             | `(3x3x10)x10`  |      `24x24x10` |      `5x5`     **CONVOLUTION BLOCK 1** [1st conv2d 3x3 layer for this block ]
` `              | `BN(10) or LN(10, 24, 24) or GN(5, 10)`   |      ` `  |      ` `
` `              | `Dropout(2.9%)`   |      ` `  |      ` `
` `              | `ReLU`   |      ` `  |      ` ` 
`24x24x10`             | `(3x3x10)x15`  |      `22x22x15` |      `7x7`     **CONVOLUTION BLOCK 1** [2nd conv2d 3x3 layer for this block ]
` `              | `BN(15) or LN(15, 22, 22) or GN(5, 15)`   |      ` `  |      ` `
` `              | `Dropout(2.9%)`   |      ` `  |      ` `
` `              | `ReLU`   |      ` `  |      ` ` 
**22x22x15**             |   **MP(2x2)**    |     **11x11x15**   |     **8x8**  **TRANSITION BLOCK 1**  [1st maxpool(2,2) layer for this block]                     
**11x11x15**             | **(1x1x15)x10**  |     **11x11x10**   |      **8x8**  **TRANSITION BLOCK 1**  [2nd 1 conv2d 1x1 layer for this block]  
** **             | **BN(10) or LN(10, 11, 11) or GN(5, 10)**   |     ** **  |     ** **                    
** **             | **Dropout(2.9%)**   |     ** **  |     ** **
** **             | **ReLU**   |     ** **  |     ** **      
*11x11x10*             | *(3x3x10)x10*  |      *9x9x10* |      *12x12*  **CONVOLUTION BLOCK 2** [1st conv2d 3x3 layer for this block]
** **            | *BN(10) or LN(10, 9, 9) or GN(5, 10)*   |     * *   |     * * 
** **             | *Dropout(2.9%)*   |     * *   |     * * 
** **             | *ReLU*   |     ** **  |    ** **                       
*9x9x10*             | *(3x3x10)x10*  |      *7x7x10* |      *16x16*  **CONVOLUTION BLOCK 2** [2nd conv2d 3x3 layer for this block] 
** **            | *BN(10) or LN(10, 7, 7) or GN(5, 10)*   |     * *   |     * * 
** **             | *Dropout(2.9%)*   |     * *   |     * * 
** **             | *ReLU*   |     ** **  |    ** **    
*7x7x10*               | *(3x3x10)x32*  |      *5x5x32*  |      *20x20*  **CONVOLUTION BLOCK 2** [3rd conv2d 3x3 layer for this block, this 32 number of output channels at this stage provides a CAPACITY BOOST for the overall network as explained ahead]  
** **            | *BN(32) or LN(32, 5, 5) or GN(4, 32)*   |     * *   |     * * 
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

##### First as a side note, on optimum values for L1-penalty-weight
    - The appropriate values for l1 was found after coarsely sweeping
      thru values from 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001 first.
    - For l1-weight, with l1-alone enabled, the 0.00001 region was found with better values
      for the validation accuracy vis-a-vis training accuracies.
    - Much finer sweeping in the 0.00001 region, led to the value: 0.000025, which seems to
      be having best accuracy values.
	  
Here is how the L1 is implemented: 	  
```python
l1_weight = 0.000025 ##Appears to be the best with the reference NW we have used here
:
:
def l1_penalty(x):
    #L1 regularization adds an L1 penalty equal
    #to the absolute value of the magnitude of coefficients
    return torch.abs(x).sum()
	:
	:
def train(model, device, train_loader, optimizer, epoch, L1=False):
    model.train()
	:
	:
    for batch_idx, (data, target) in enumerate(pbar):
        
		:	
		:
        # Predict
        y_pred = model(data)
        if L1:
            to_reg = []
            for param in model.parameters():
                to_reg.append(param.view(-1))
            l1 = l1_weight*l1_penalty(torch.cat(to_reg))
        else:
            l1 = 0
        # Calculate loss
        #L1 regularization adds an L1 penalty equal to the 
        #absolute value of the magnitude of coefficients
        loss = F.nll_loss(y_pred, target) + l1     
        :
		:

 ```


###  2 graphs to show the validation accuracy change and loss change:

Consolidated validation accuracy plot for 3 iterations:

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_5/model_consolidated_test_acc.png "Logo Title Text 1")

Consolidated validation loss plot for 3 iterations:

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_5/model_consolidated_test_loss.png "Logo Title Text 1")

###  Misclassified images:

For Best saved Model (MNIST_model_epoch-20_L1-0_group-Norm_val_acc-99.41) with Group Normalization alone:

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_5/misclassified_grp_norm_l1_0.png "Logo Title Text 1")


For Best saved Model (MNIST_model_epoch-18_L1-0_layer-Norm_val_acc-99.37) with Layer Normalization alone:

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_5/misclassified_layer_norm_l1_0.png "Logo Title Text 1")


For Best saved Model (MNIST_model_epoch-19_L1-1_batch-Norm_val_acc-99.47) with Batch Normalization & L1:

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_5/misclassified_batch_norm_l1_1.png "Logo Title Text 1")


### How to calculate  µ & σ², while performing the 3 normalizations techniques (BN/LN/GN):

Please refer the [xls sheet](https://github.com/ojhajayant/EVA8/blob/main/session_5/Normalizations.xlsx) which has calculations on batch, layer & group normalizations.
This sheet shows an Nth Layer (which has in turn 4 channels of 3x3 size, as an example) for a mini-batch size of 4 images.Overall, during the process of either of batch,layer or group normalizations we want to know, across what pixel values are we calculating our mean and variances.Following snaps below try to illustrate that itself.

The snapshot just below represents all the calculated values for the mean and variance under batch, layer & group normalizations.Please note that as the randinbetween(-3, 3) formula, used in the 3x3 channel pixel values, changes some values shown during the following steps during clicks on various cells, but overall for any given snapshot in following diagrams its all correct (which can in turn be verified in the uploaded [xls sheet](https://github.com/ojhajayant/EVA8/blob/main/session_5/Normalizations.xlsx) too.)


![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_5/all_normalization_calculated.png "Logo Title Text 1")

The snapshot below represents, how for batch normalization, one of the mean values, for one of the channels out of 4 across different batch images is calculated.Same can be extended to mean values for the remaining 3 more channels.Hence for batch-normalization here, we end up having 4 mean values for 4 channels.

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_5/one_sample_mean_under_batch_norm_calculated.png "Logo Title Text 1")

The snapshot below represents, how for batch normalization, one of the variance values, for one of the channels out of 4 across different batch images is calculated.Same can be extended to variance values for the remaining 3 more channels.Hence for batch-normalization here, we end up having 4 variance values for 4 channels.

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_5/one_sample_var_under_batch_norm_calculated.png "Logo Title Text 1")

The snapshot below represents, how for layer normalization, one of the mean values is calculated for all the 4 channels of Nth layer, for a single mini-batch sample. Same can be extended to remaining 3 mini-batch samples.Hence for layer-normalization here, we end up having 4 mean values for a batch size of 4 images.

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_5/one_sample_mean_under_layer_norm_calculated.png "Logo Title Text 1")

The snapshot below represents, how for layer normalization, one of the variance values is calculated for all the 4 channels of Nth layer, for a single mini-batch sample. Same can be extended to remaining 3 mini-batch samples.Hence for layer-normalization here, we end up having 4 variance values for a batch size of 4 images.

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_5/one_sample_var_under_layer_norm_calculated.png "Logo Title Text 1")

The snapshot below represents the calculation for group normalization, where as an example, the 4 channels under different mini-batch image samples are divided under 2 groups.The snapshot below, specifically shows how the mean is calculated for the first mini-batch sample under the 1st such group of channels. Same can be extended to the first groups for remaining 3 mini batch image samples too.

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_5/one_sample_mean_for_grp_1_under_group_norm_calculated.png "Logo Title Text 1")

The snapshot below represents the calculation for group normalization, where as an example, the 4 channels under different mini-batch image samples are divided under 2 groups.The snapshot below, specifically shows how the mean is calculated for the first mini-batch sample under the 2nd such group of channels. Same can be extended to the 2nd groups for remaining 3 mini batch image samples too.

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_5/one_sample_mean_for_grp_2_under_group_norm_calculated.png "Logo Title Text 1")

The snapshot below represents the calculation for group normalization, where as an example, the 4 channels under different mini-batch image samples are divided under 2 groups.The snapshot below, specifically shows how the variance is calculated for the first mini-batch sample under the 1st such group of channels. Same can be extended to the first groups for remaining 3 mini batch image samples too.

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_5/one_sample_var_for_grp_1_under_group_norm_calculated.png "Logo Title Text 1")

The snapshot below represents the calculation for group normalization, where as an example, the 4 channels under different mini-batch image samples are divided under 2 groups.The snapshot below, specifically shows how the variance is calculated for the first mini-batch sample under the 2nd such group of channels. Same can be extended to the 2nd groups for remaining 3 mini batch image samples too.

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_5/one_sample_var_for_grp_2_under_group_norm_calculated.png "Logo Title Text 1")

###  Confusion Matrix & Classification Reports:

For Best saved Model (MNIST_model_epoch-20_L1-0_group-Norm_val_acc-99.41) with Group Normalization alone:
![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_5/confusion_matrix%20%26%20classification_rpt_grp_norm_l1_0.png "Logo Title Text 1")

For Best saved Model (MNIST_model_epoch-18_L1-0_layer-Norm_val_acc-99.37) with Layer Normalization alone:

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_5/confusion_matrix%20%26%20classification_rpt_layer_norm_l1_0.png "Logo Title Text 1")

For Best saved Model (MNIST_model_epoch-19_L1-1_batch-Norm_val_acc-99.47) with Batch Normalization & L1:

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_5/confusion_matrix%20%26%20classification_rpt_batch_norm_l1_1.png "Logo Title Text 1")



	- First observation: the parameters under Layer Normaliztion option shot up to 53,670 unlike 
	  the Group & Batch normalization options (7,684 each)
	- The logs and accuracy plot comparisons put the L1+BN option as best one(~99.47%) as compared to the
	  other two, while the one with Layer norm fared worst (~99.37%) among these.
	
















