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
from code_RNN import ELM
from einops.layers.torch import Rearrange


POOLINGS = {'AvgPool2d': nn.AdaptiveAvgPool2d((1,1)), #Global Average Pooling (GAP)
            'RAEspatial': lambda M: randomized_encoding(M, dim_norm=(1,2)),
            'RAEspectral': lambda M: randomized_encoding(M, dim_norm=0),
            'RAEflat': lambda M: randomized_encoding(M, dim_norm=(0,1,2))
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
    def __init__(self, model, output_stride=True):
        super(timm_feature_extractor, self).__init__() 
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
    
    #The forward pass consists of input 4d tensor, pooling method, and M is only used for the ELM method
    def forward(self, x, depth='last', pooling='AvgPool2d', M=1): 
        self.net.eval() #this is needed to disable batch norm, dropout, etc
        fmap = self.net(x)
        
        #the depth of the features is selected according to dict FEATURE_DEPTH
        #   it accepts either a single depth (eg. last, middle, etc)
        #   or 'all', meaning the aggregative method: all feature blocs returned by timm are concatenated
        if depth != 'all': 
            fmap = fmap[int(FEATURE_DEPTH[depth](len(fmap)))]
            
        if 'RAE' in pooling:
            return torch.flatten(POOLINGS[pooling](M)(fmap),
                             start_dim=1, end_dim=-1)
        elif depth=='all':
            return torch.flatten(aggregation_GAP()(fmap),
                                  start_dim=1, end_dim=-1)
        else:
            return torch.flatten(POOLINGS[pooling](fmap),
                                  start_dim=1, end_dim=-1)
  
class randomized_encoding(Module):
    def __init__(self, M, dim_norm=(1,2)):
        super(randomized_encoding, self).__init__()         
        self.M = M
        self.dim_=dim_norm
        
    def forward(self, x):
        ###### PARAMETERIZATION
        #   The parameterization can be moved to the class constructor if all the sizes of the activation maps are known
        #   we decided to keep it here for the code to be more generic and adaptative to different backbones, data, etc
        
        #this is the case when a single activation map is given, therefore
        if not isinstance(x, list): 
            x = [x] #just make x as a list, to loop below
            SPATIAL_ = x[0].size()[2] #the spatial dimms will simply be the dimm of the single activation map
        else:
            SPATIAL_ = x[int(np.round(len(x)/2))].size()[2] 
            
        func = Rearrange('c h w -> c (h w)') #einops function for dimm reorganization
        batch, z, _, _ = x[0].size() #zero is the first activation map
        _, zn, _, _ = x[-1].size() #-1 is the last activation map
        Q=1 #we tried increasing Q, but simply summing weights is not a good approach in this case 
        M = self.M #ensemble size. M randomized AEs with different seed are summed after training        
        device = x[0].get_device()
        out = []
        P = sum([a.size()[1] for a in x])  
        ELMs = []
        for model in range(M):
            ELMs.append(ELM.ELM_AE(Q=Q, P=P, N=SPATIAL_**2, device=x[0].get_device(), seed=model*(Q*P)))
        ##### PARAMETERIZATION ends ####################################################################################     
       
        for sample in range(batch):            
            meta_activation_map = torch.empty((0, SPATIAL_**2)).to(device) #this will be the final tensor to be processed              
            for depth in range(len(x)):    
                _norm('l2', x[depth][sample], dim=self.dim_) 
                # _norm('l2', x[depth][sample], dim=(1,2))
                #stack activation maps accross z, for all layers used for feature extraction
                meta_activation_map = torch.cat((meta_activation_map, 
                                              func(torchvision.transforms.Resize(SPATIAL_)(x[depth][sample]))), axis=0)                
            
            #train each of the M ELMs and summ their output weights to compose the features    
            features = torch.zeros(Q,P).to(device)
            for elm in ELMs:
                features += elm.fit_AE(meta_activation_map)         
            
            #if more then 1 hidden neuron is used, the weights are additionally summed
            #accross the hidden dimension, i.e., the weights of each output neuron, resulting 
            #in 1 value for each output neuron = number of input features = sum(z_i) for considered i
            if Q==1:
                pooled_features=torch.reshape(features, (1, features.size()[0]*features.size()[1]))
            else:
                pooled_features = torch.sum(features, axis=0)
            
            #to avoid rare cases when some features are nan or +-inf, due to signal exploding 
            #   in the forward pass of the backbone, causing extreme values to be used to train the RAE
            pooled_features= torch.nan_to_num(pooled_features) 
            out.append(pooled_features)

        return torch.stack(out)


def _norm(method, data, dim):
    if method == 'l2':
        torch.nn.functional.normalize(data, p=2.0, dim=dim, eps=1e-10, out=data)
        
    elif method == 'zscore' and dim == (1,2):
        z,h,w = data.size()
        avg =  torch.reshape(torch.mean(data, dim=dim), (z,1,1)) #z-sized vector
        var =  torch.reshape(torch.add(torch.std(data, dim=dim), 1e-10), (z,1,1))        
        torch.divide(torch.subtract(data, avg), var, out=data)
               
    else:
        torch.divide(torch.subtract(data, torch.mean(data, dim=dim)),
                     torch.std(data, dim=dim), out=data)
        
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

    