# Session_6

### 

> Run this network: https://colab.research.google.com/drive/1qlewMtxcAJT6fIJdmMh8pSf2e-dh51Rw

> Fix the network above:

*   change the code such that it uses GPU and
*   change the architecture to C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, extra credits!)
*   total RF must be more than 44
*   one of the layers must use Depthwise Separable Convolution
*   one of the layers must use Dilated Convolution
*   use GAP (compulsory):- add FC after GAP to target #of classes (optional)
*   use albumentation library and apply:

   > horizontal flip

   > shiftScaleRotate

   > coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
*   achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.

Please refer the [notebook](https://github.com/ojhajayant/EVA8/blob/main/session_6/EVA8_session6_assignment.ipynb) for this assignment solution.

#### Few details on the Model with batch/group/layer norm options:

Please refer [model.py](https://github.com/ojhajayant/EVA8/blob/main/session_6/modular/models/model.py)

First of all, for importing the single model file named "model.py" under which model class named "EVA8_session6_assignment_model" is defined.Also different other components have been now made modular in nature.

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_6/modular_struct.png "Logo Title Text 1")

```python
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Collecting git+https://****@github.com/ojhajayant/EVA8.git
  Cloning https://****@github.com/ojhajayant/EVA8.git to /tmp/pip-req-build-_qfd7otz
  Running command git clone --filter=blob:none --quiet 'https://****@github.com/ojhajayant/EVA8.git' /tmp/pip-req-build-_qfd7otz
  Resolved https://****@github.com/ojhajayant/EVA8.git to commit 46f52015e1e9128a2d0c070be90b57d5e4d0f7a0
  Preparing metadata (setup.py) ... done
Building wheels for collected packages: eva8
  Building wheel for eva8 (setup.py) ... done
  Created wheel for eva8: filename=eva8-0.0.0-py3-none-any.whl size=17727 sha256=765cceb8cbd243c03daa79b495d5c3ed20d649d1c759ba6e5c6f51c2d86916bf
  Stored in directory: /tmp/pip-ephem-wheel-cache-2ktyjweq/wheels/91/2b/81/6528ca90c705fbe7f126306baa4c34cd2cbf0ea8fcb5a3dd90
Successfully built eva8
Installing collected packages: eva8
Successfully installed eva8-0.0.0
 ```


   
######    Please Note that this model has been structured as:
                            C1-T1-C2-T2-C3-T3-C4-O
							
    > C1, C2 & C3 BLOCKS:
	
    The underlying structure, each of the C1, C2, C3 (i.e. the conv2d blocks,
    which are followed by a transition layer are made up of 2 conv2d layers:
	
        > first one-- a normal/usual 3x3 Conv2d with padding=1 & stride 1
        &
        > second one--a dilated kernel, chosen such that it emulates full-fledged
                    max-pool function under C1 & C2 blocks (i.e. the effective
                    kernel size is 5x5(by having dilation rate =2, padding=2
                    & stride=2, thus under C1 & C2 block positions this layer
                    converts an input of 32x32 to 16x16 @ C1 and 16x16 to
                    8x8 @ C2), But under C3 this though has the effective kernel
                    size still at 5x5 (using dilation rate value=2) but the
                    padding and stride values both have been changed to 1.
                    Thus, for the C3 position, rather than using a perfect
                    max-pool function (which would have changed the input size
                    of 8x8 to 4x4, we instead with this careful choice of
                    effective kernel size =5, padding =1 & stride=1 for this
                    layer, under C3 block position, convert the input-size from
                    8x8 to an output of 6x6 instead (as for the GAP stage & also
                    for any sort of grad-cam assisted class mapping an output
                    size less than 5x5 is not desirable)
					
    > TRANSITION BLOCKS (1, 2 & 3) :
	
    While all the three transition layers have a normal/usual 1x1 conv2d with
    padding = 0 & stride = 1 (actually these work along with the 2nd layer
    of the C1, C2 & C3 layers as explained earlier.)
	
    > C4 BLOCK: 
	
	a "capacity boosting" number required  for the output layer (here
    though, the chosen 128 value, is same for earlier layers)
	
    > OUTPUT BLOCK: 
	
	GAP layer followed by a 1x1 operator (which actually
    resembles a fully-connected(FC) layer in this case. A noteworthy point (
    which relates to the "capacity" element of the overall network (NW)) is
    that the FC (the 1x1 conv2d behaviour here) will work better,
    in generating the final 10 "class-features" to be used by the
    log_softmax.If the inputs to it, have more "features points", i.e. for
    example if we do a 64->10 conversion, vs a  conversion, we can
    expect the 10 class-features (for softmax to decide) generated by a
    128->10 conversion will be more "robust" (as compared to say a 64->10
    conversion)
    Also note a choice of 256->10 can prove to be prone to overfitting, hence
    128->10 choice is optimum.
	
	
    Overall, in terms of the receptive field we get an RF of 71x71 for this
    design.Which is greater than the minimum requirement of 44x44. 
	



###  Following graph shows the model accuracy:

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_6/model_acc.png "Logo Title Text 1")

###  Misclassified images:


For Best saved Model :

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_6/misclassified_samples.png "Logo Title Text 1")



###  Confusion Matrix & Classification Reports:

For Best saved Model:

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_6/confusion_matrix_classification_report.png "Logo Title Text 1")















