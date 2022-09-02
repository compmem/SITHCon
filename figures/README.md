<h2 align="center">
<a href="https://proceedings.mlr.press/v162/jacques22a.html">A deep convolutional neural network that is invariant to time rescaling</a>
</h2>

<h4 align="center">
  <a href="#overview">Overview</a> |
  <a href="#installation">Installation</a> |
  <a href="#deepsith-use">SITHCon</a> |
  <a href="#examples">Examples</a>  
</h4>


## Overview

![SITHConLayout](/figures/model_layout.png)

Here we present a deep Scale-Invariant Temporal History Convolution network (SITHCon) that uses a logarithmically
compressed temporal representation at each level. Because time rescaling of the input results in a
translation of the memory representation over log time, and because the output of the convolution is
equivariant to translations, this network can generalize to out-of-sample data that are temporal
rescalings of a learned pattern. We compare the performance of SITHCon to a Temporal Convo-
lution Network (TCN) on classification and regression problems with both univariate and mul-
tivariate time series. We find that SITHCon, unlike TCN, generalizes robustly over rescalings of
about an order of magnitude. Moreover, we show that the network can generalize over exponentially
large scales without retraining the weights simply by extending the range of the logarithmically-
compressed temporal memory.

## Installation

The easiest way to install the SITHCon module is with pip.

    pip install .
    
### Requirements

SITHCon requires at least PyTorch 1.8.1. It works with cuda, so please follow the instructions for installing pytorch and cuda <a href="https://pytorch.org/get-started/locally/">here</a>.

## SITHCon
SITHCon is a pytorch module implementing the neurally inspired SITH representation of working memory for use in neural networks. 
The paper outlining the work detailed in this repository was published at ICML 2022 <a href="https://proceedings.mlr.press/v162/jacques22a/jacques22a.pdf">here</a>. 

Jacques, B., Tiganj, Z., Sarkar, Aakash, Howard, M., &amp; Sederberg, P. (2021, December 6). A deep convolutional neural network that is invariant to time rescaling

The SITHCon layer effeciently detects patterns in timeseries data by training a convolutional layer to be applied on \emph{compressed time} instead of 
actual time. SITHCon layers will first transform input timeseries into a compressed time representation at every time point, creating a set of nodes that
evolve at different speeds through time. Some nodes code for the recent past and these nodes also evolve extremely quickly through time. Some nodes code for 
the distant past of a timeseries, and these nodes evolve more slowly and integrate information from a much wider amount of the past. 

This compressed representation is equivariant to changes in scale, meaning if you change the scale of the input being presented, the compressed time 
representation will translate across the nodes but maintain the same shape. When combined with a convoltuional layer and a maxpooling layer, you generate
scale-invariant features as the output for the SITHCon Layer.

### How to use

The SITHCon module in pytorch will initialize as a series of SITHCon layers, parameterized by the argument `layer_params`, which is a list of dictionaries. Below is an example initializing a 2 layer SITHCon module, where the input time-series only has 1 feature. 

    from sithcon import SITHCon_Classifier
    from torch import nn as nn
    
    # Tensor Type. Use torch.cuda.FloatTensor to put all SITH math 
    # on the GPU.
    ttype = torch.FloatTensors
    
    sp1 = dict(in_features=1, 
               tau_min=.1, tau_max=4000, buff_max=6500,
               dt=1, ntau=400, k=35, g=0.0, ttype=ttype, 
               channels=35, kernel_width=23, dilation=2,
               dropout=None, batch_norm=None)
    sp2 = dict(in_features=sp1['channels'], 
               tau_min=.1, tau_max=4000, buff_max=6500,
               dt=1, ntau=400, k=35, g=0.0, ttype=ttype, 
               channels=35, kernel_width=23, dilation=2, 
               dropout=None, batch_norm=None)

    # TWO LAYERS
    layer_params = [sp1, sp2]
    
    # Predicting 10 classes
    model = SITHCon_Classifier(10, layer_params, act_func=nn.ReLU).cuda()

Here, we have the first layer only having 15 taustar from `tau_min=1.0` to `tau_max=25`. The second layer is set up to go from `1.0` to `100.0`, which gives it 4 times the temporal range. We found that the logarithmic increase of layer sizes to work well for the experiments in this repository. 

The SITHCon_Classifier module expects an input signal of size (batch_size, 1, sith_params1["in_features"], Time). 

If you want to use **only** the SITH module, which is a part of any SITHCon layer, you can initialize a SITH using the following parameters. Note, these parameters are also used in the dictionaries above.

#### SITH Parameters
- tau_min: float
    The center of the temporal receptive field for the first taustar produced. 
- tau_max: float
    The center of the temporal receptive field for the last taustar produced. 
- buff_max: int
    The maximum time in which the filters go into the past. NOTE: In order to 
    achieve as few edge effects as possible, buff_max needs to be bigger than
    tau_max, and dependent on k, such that the filters have enough time to reach 
    very close to 0.0. Plot the filters and you will see them go to 0. 
- k: int
    Temporal Specificity of the taustars. If this number is high, then taustars
    will always be more narrow.
- ntau: int
    Number of taustars produced, spread out logarithmically.
- dt: float
    The time delta of the model. There will be int(buff_max/dt) filters per
    taustar. Essentially this is the base rate of information being presented to the model
- g: float
    Typically between 0 and 1. This parameter is the scaling factor of the output
    of the module. If set to 1, the output amplitude for a delta function will be
    identical through time. If set to 0, the amplitude will decay into the past, 
    getting smaller and smaller. This value should be picked on an application to 
    application basis.
- ttype: Torch Tensor
    This is the type we set the internal mechanism of the model to before running. 
    In order to calculate the filters, we must use a DoubleTensor, but this is no 
    longer necessary after they are calculated. By default we set the filters to 
    be FloatTensors. NOTE: If you plan to use CUDA, you need to pass in a 
    cuda.FloatTensor as the ttype, as using .cuda() will not put these filters on 
    the gpu. 

Initializing SITH will generate several attributes that depend heavily on the values of the parameters. 

- c: float
    `c = (tau_max/tau_min)**(1./(ntau-1))-1`. This is the description of how the distance between
    taustars evolves. 
- tau_star: DoubleTensor
    `tau_star = tau_min*(1+c)**torch.arange(ntau)`. This is the array filled with all of the
    centers of all the tau_star receptive fields. 
- filters: ttype
    The generated convolutional filters to generate SITH output. Will be applied as a convolution
    to the input time-series.


## Examples

In the `experiments` folder are the experiments that were included in the paper. Everything to recreate the results therein is included. Everything is in jupyter notebooks. We have also included everything needed to recreate the figures from the paper, but with your results if you change file names around. 


