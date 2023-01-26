# Session_5

# Make 3 versions of 4th Session assignment's best model :
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


Please refer the [notebook]([https://github](https://github.com/ojhajayant/EVA8/blob/main/session_5/EVA8_session5_assignment.ipynb)) for this assignment solution.

#### Model with batch/group/layer norm options:

Please refer the [model.py] (https://github.com/ojhajayant/EVA8/blob/main/session_5/modular/models/model.py)

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

##### First as a side note, on optimum values for L1-penalty-weight
    - The appropriate values for l1 was found after coarsely sweeping
      thru values from 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001 first.
    - For l1-weight, with l1-alone enabled, the 0.00001 region was found with better values
      for the validation accuracy vis-a-vis training accuracies.
    - Much finer sweeping in the 0.00001 region, led to the value: 0.000025, which seems to
      be having best accuracy values.
	  
Here is how the L1 is implemented: 	  
```python

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
![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_5/model_consolidated_test_acc.png "Logo Title Text 1")

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_5/model_consolidated_test_loss.png "Logo Title Text 1")

###  misclassified images:
![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_5/misclassified_grp_norm_l1_0.png "Logo Title Text 1")
![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_5/misclassified_layer_norm_l1_0.png "Logo Title Text 1")
![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_5/misclassified_batch_norm_l1_1.png "Logo Title Text 1")


Please refer the [xls sheet]([https://github.com/ojhajayant/EVA8/blob/main/session_3/BackPropogationCalculation.xlsx](https://github.com/ojhajayant/EVA8/blob/main/session_5/Normalizations.xlsx)) which has calculations on batch, layer & group normalizations.

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_5/all_normalization_calculated.png "Logo Title Text 1")

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_5/one_sample_mean_under_batch_norm_calculated.png "Logo Title Text 1")

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_5/one_sample_var_under_batch_norm_calculated.png "Logo Title Text 1")

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_5/one_sample_mean_under_layer_norm_calculated.png "Logo Title Text 1")

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_5/one_sample_var_under_layer_norm_calculated.png "Logo Title Text 1")

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_5/one_sample_mean_for_grp_1_under_group_norm_calculated.png "Logo Title Text 1")

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_5/one_sample_mean_for_grp_2_under_group_norm_calculated.png "Logo Title Text 1")

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_5/one_sample_var_for_grp_1_under_group_norm_calculated.png "Logo Title Text 1")

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_5/one_sample_var_for_grp_2_under_group_norm_calculated.png "Logo Title Text 1")















