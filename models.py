# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 14:29:20 2022

Imported and/or created torch/timm models

@author: scabini
"""
import torch
import torchvision
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
# from scipy import stats
import timm
import numpy as np
# from scipy import io
# from code_RNN import RNN
from RNN import RAE
from einops.layers.torch import Rearrange


POOLINGS = {'AvgPool2d': nn.AdaptiveAvgPool2d((1,1)), #Global Average Pooling (GAP)
            'RAEspatial':  lambda spatial_dims, z, M, device: RADAM(spatial_dims, z, M, dim_norm=(2,3), device=device, pos_encoding=True),
            'RAEspatialnoembed':  lambda spatial_dims, z, M, device: RADAM(spatial_dims, z, M, dim_norm=(2,3), device=device, pos_encoding=False), #this one needs manual changin in ELM.py
            'RAEspectral': lambda spatial_dims, z, M, device: RADAM(spatial_dims, z, M, dim_norm=1, device=device),
            'RAEflat':     lambda spatial_dims, z, M, device: RADAM(spatial_dims, z, M, dim_norm=(1,2,3), device=device)
            # 'Spectral_Conv' : lambda WH : Spectral_Conv(1)
            }


FEATURE_DEPTH = {'all' : lambda d: '!', #this case is treated with an if, see below
                 'last': lambda d: -1,
                 'top_quarter': lambda d: int(round((d/4)*3)-1) if d%2 == 0 else d//4*3, #around 75% of the NN depth
                 'middle': lambda d: int(round((d/2))-1) if d%2 == 0 else d//2, #returns an index around 50% of the NN depth
                 'quarter': lambda d: int(round((d/4))-1) if d%2 == 0 else d//4} #around 25% of the NN depth

class Object(object):
    pass

class Accuracy(nn.Module):
    def forward(self, x, y):
        y_pred = F.softmax(x, dim=1).argmax(dim=1).cpu().numpy()
        y = y.cpu().numpy()
        return accuracy_score(y_true=y, y_pred=y_pred)
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

 
class timm_feature_extractor(Module):    
    def __init__(self, model, output_stride=True, input_dim=None, 
                 depth='all', pooling='RAEspatial', M=4, device="cuda" if torch.cuda.is_available() else "cpu"):
        
        super(timm_feature_extractor, self).__init__() 
        self.depth=depth
        # We try to use output_strides to control the spatial dims at different depths 
        #   using timm builtin functions. If this is not possible, or if just some of the
        #   depths are resized, torchvision.transforms.Resize is used by our RADAM module below
        #dilate spatial sizes of activation maps?
        if output_stride==False: #if not, standard spatial dimms are kept
            self.net = timm.create_model(model, features_only=True, pretrained=True)
        else:# if true, then tries to use some strides (different models has different working strides)
            try:
                self.net = timm.create_model(model, features_only=True, pretrained=True, output_stride=8)
            except:
                try:
                    self.net = timm.create_model(model, features_only=True, pretrained=True, output_stride=32)
                except: #if it is not possible to increase spatial dimms, return standard dimms
                    self.net = timm.create_model(model, features_only=True, pretrained=True)
         
        #disable training for the backbone: the model is used only as feature extraction
        for param in self.net.parameters():
            param.requires_grad = False           
        self.net.eval()
        
        #this is the case when the input dim is not known or images with multiple
        #   resolutions are being used. It will be less efficient since RADAM
        #   will need to initialize the RAE weights at every forward pass
        if input_dim==None:
            if 'RAE' in pooling:
                TODO=None
            elif depth=='all':
                self.pooler = aggregation_GAP()
            else:
                self.pooler =  POOLINGS[pooling]
            
        #if input dim is given then the size of the RAEs of RADAM will always be
        #   the same, so we initialize it only once.
        #   ATTENTION! The current implementation only works with squared inputs.
        #   For non-square inputs, check SPATIAL_ variable, the RADAM initialization and also positional encoding
        else:                            
            if 'RAE' in pooling:
                #using a dummy tensor to check the sizes of activation maps for the given backbone
                fmap_init = self.net(torch.zeros(1,3,input_dim,input_dim)) 
                
                if depth != 'all': 
                    fmap_init = fmap_init[int(FEATURE_DEPTH[depth](len(fmap_init)))]
                    SPATIAL_ = fmap_init[0].size()[2] #the spatial dims will simply be the dimm of the single activation map
                    z=fmap_init[0].size()[1]
                else:
                    #aggregative activation map: select the spatial dims around the middle of the list, considering how
                    #   timm's features_only=True returns the activation blocks
                    SPATIAL_ = fmap_init[int(np.round(len(fmap_init)/2))].size()[2]
                    z = sum([a.size()[1] for a in fmap_init]) 
                    
                self.pooler =  POOLINGS[pooling](SPATIAL_, z, M, device)
            elif depth=='all':
                self.pooler = aggregation_GAP()
            else:
                self.pooler =  POOLINGS[pooling]

    #The forward pass consists of input 4d tensor, pooling method, and M is only used for the RADAM method
    def forward(self, x): 
        self.net.eval() #this is needed to disable batch norm, dropout, etc
        fmap = self.net(x)
        
        #the depth of the features is selected according to dict FEATURE_DEPTH
        #   it accepts either a single depth (eg. last, middle, etc)
        #   or 'all', meaning the aggregative method: all feature blocks returned by timm are concatenated        
        if self.depth != 'all': 
            fmap = fmap[int(FEATURE_DEPTH[self.depth](len(fmap)))]
            
        return torch.flatten(self.pooler(fmap),
                             start_dim=1, end_dim=-1)
       
  
class RADAM(Module):
    #The current implementation does not fully integrate RADAM into the backbone, 
    #   so we need to set some things manually like the device to put the RAEs
    def __init__(self, spatial_dims, z, M, device, pos_encoding, dim_norm=(2,3)):
        super(RADAM, self).__init__()         
        # self.dim_=dim_norm
        self.SPATIAL_ = spatial_dims

    ###### RADAM PARAMETERIZATION
        #   The parameterization is kept in the class constructor for better efficiency.
        #   It could be moved to the forward pass if the input dims are unknown.       
        
        self.rearange = nn.Sequential(lp_norm_layer(p=2.0, dim=dim_norm, eps=1e-10),
                                      torchvision.transforms.Resize(self.SPATIAL_),
                                      Rearrange('b c h w -> b c (h w)')
                                     )  #einops function for dimm reorganization
        self.Q=1 #we tried increasing Q, but simply summing weights is not a good approach in this case 
        self.z=z
        self.device = device

        self.RAEs = []
        for model in range(M):
            self.RAEs.append(RAE(Q=self.Q, P=z, N=self.SPATIAL_**2, device=device, seed=model*(self.Q*z), pos_encoding=pos_encoding))
    ##### PARAMETERIZATION ends #################################################################################### 
        
    def forward(self, x):        
        out = []
        #this is the case when a single activation map is given, therefore
        if not isinstance(x, list): 
            x = [x] #just make x as a list, to loop below
            
        device = self.device
        batch, _, _, _ = x[0].size()

        #Our current implementation of RAEs/RADAM does not support batch computation.
        #   therefore, we need to loop over the batch's samples and each of their activation maps
        # wh = self.SPATIAL_**2
        
        for depth in range(len(x)): 
            x[depth] = self.rearange(x[depth]) 
            
        for sample in range(batch):            
            meta_activation_map = torch.vstack([x[i][sample] for i in range(len(x))])           
            
            #train each of the m RAEs with leas-squares (see ELM.py) and summ their decoder weights 
            pooled_features = torch.zeros(self.Q,self.z).to(device)
            
            for rae in self.RAEs:
                pooled_features += rae.fit_AE(meta_activation_map)  
                
            #if more then 1 hidden neuron is used, the weights are additionally summed
            #accross the hidden dimension, i.e., the weights of each output neuron, resulting 
            #in 1 value for each output neuron = number of input features = sum(z_i) for considered i
            # if self.Q==1:
            #     pooled_features=torch.reshape(features, (1, features.size()[0]*features.size()[1]))
            # else:
            #     pooled_features = torch.sum(features, axis=0)
            
            #to avoid rare cases when some features are nan or +-inf, due to signal exploding 
            #   in the forward pass of the backbone, causing extreme values to be used to train the RAE
            pooled_features= torch.nan_to_num(pooled_features) 
            out.append(pooled_features)

        return torch.stack(out)


class lp_norm_layer(nn.Module):
    """ """
    def __init__(self,p=2.0, dim=(1,2), eps=1e-10):
        super().__init__()   
        self.p=p
        self.dim=dim
        self.eps=eps

    def forward(self, x):
        return torch.nn.functional.normalize(x, p=self.p, dim=self.dim, eps=self.eps)
    
# def _norm(method, data, dim):
#     if method == 'l2':
#         torch.nn.functional.normalize(data, p=2.0, dim=dim, eps=1e-10, out=data)
        
#     elif method == 'zscore' and dim == (1,2):
#         z,h,w = data.size()
#         avg =  torch.reshape(torch.mean(data, dim=dim), (z,1,1)) #z-sized vector
#         var =  torch.reshape(torch.add(torch.std(data, dim=dim), 1e-10), (z,1,1))        
#         torch.divide(torch.subtract(data, avg), var, out=data)
               
#     else:
#         torch.divide(torch.subtract(data, torch.mean(data, dim=dim)),
#                      torch.std(data, dim=dim), out=data)
        
class aggregation_GAP(Module):
    def __init__(self):
        super(aggregation_GAP, self).__init__()        
    
    def forward(self, x):
        batch, z, w, h = x[0].size()
              
        meta_feature_maps = torch.empty((batch, 0)).to(x[0].get_device())
        for ii in range(len(x)): #do GAP at each depth
            meta_feature_maps = torch.cat((meta_feature_maps,
                                           torch.flatten(nn.AdaptiveAvgPool2d((1,1))(x[ii]),
                                             start_dim=1, end_dim=-1)), axis=1)
            
        return meta_feature_maps 

    