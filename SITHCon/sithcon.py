# Layers of Laplace Transforms pytorch Layer
# PyTorch version 0.1.0
# Authors: Brandon G. Jacques and Per B. Sederberg
import torch
from torch import nn 
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from .isith import iSITH

class _TCTCT_Core(nn.Module):
    """"""
    def __init__(self, layer_params):
        super(_TCTCT_Core, self).__init__()
        self.in_features = layer_params.pop('in_features', None)
        self.channels = layer_params.pop('channels', 5)
        self.kernel_width = layer_params.pop('kernel_width', 5)
        self.dilation = layer_params.pop('dilation', 1)
        

        assert(self.in_features)
        
        self.sith = iSITH(**layer_params)
        ntau = self.sith.ntau
        assert(self.kernel_width <= ntau)
        
        # Bias = False for good reason
        # Wrapped in a weight norm (seemed to give better resluts)
        self.conv = weight_norm(nn.Conv2d(1, self.channels,
                                          kernel_size=(self.in_features, 
                                                       self.kernel_width), 
                                          dilation=(1, self.dilation), bias=False)) 

        self.maxp = nn.MaxPool1d(ntau - self.dilation*(self.kernel_width - 1))
        
        
        
        # initialize the weights
        nn.init.kaiming_normal_(self.conv.weight.data)

    def forward(self, inp):
        # Outputs as : [Batch, features, tau, sequence]
        x = self.sith(inp)
        # Swap sequence and features 
        x = x.transpose(3,1)

        x = self.conv(x.reshape(-1, 1, x.shape[2], x.shape[3]))
        
        x = self.maxp(x.squeeze(2))

        return x

    
class TCTCT_Layer(nn.Module):
    """"""
    def __init__(self, layer_params, act_func=None, 
                 dropout=.2, batch_norm = True):

        super(TCTCT_Layer, self).__init__()
        
        self.tctct = _TCTCT_Core(layer_params)
        
        if act_func:
            self.act_func = act_func()
        else:
            self.act_func = None
            
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(self.tctct.channels)
        else:
            self.batch_norm = None
            
        self.dropout = nn.Dropout(p=dropout)
        
    
    def forward(self, inp):
        batch_size = inp.shape[0]
        seq_size = inp.shape[-1]
        
        inp = self.tctct(inp)
        # back to batches, sequences, features
        inp = inp.reshape(batch_size, seq_size, -1)
        
        if self.act_func:
            inp = self.act_func(inp)
            
        if self.batch_norm:
            inp = inp.transpose(2,1)
            inp = self.batch_norm(inp).transpose(2,1)
            
        inp = self.dropout(inp)

        # then to a useful view for the next layer
        inp = inp.transpose(2,1).unsqueeze(1)
        
        return inp
        
        