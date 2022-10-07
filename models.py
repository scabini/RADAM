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
from scipy import stats
import timm
import numpy as np
from scipy import io
from code_RNN import RNN
from code_RNN import ELM, ELM_conv, RATT
from einops.layers.torch import Rearrange


POOLINGS = {'AvgPool2d': lambda WH: nn.AdaptiveAvgPool2d((1,1)),
            'MaxPool2d': lambda WH: nn.MaxPool2d(WH),
            # 'AdaptiveAvgPool2d': lambda WH: nn.AdaptiveAvgPool2d(int(1)), #same as standard pooling when doing global pooling
            # 'AdaptiveMaxPool2d': lambda WH: nn.AdaptiveMaxPool2d(int(1)),
            'FractionalMaxPool2d': lambda WH: nn.FractionalMaxPool2d(WH, output_size=1),
            'LPPool2d': lambda WH: nn.LPPool2d(2, WH),
            'ELMspatial': lambda WH: ELM_AE_Spatial_Pool2d(WH),
            'ELMspectral': lambda WH: ELM_AE_Spectral_Pool2d(WH),
            # 'ELM_spectral_torch': lambda WH: ELM_AE_Spectral_Pool2d_torch(WH),
            'ELMfatspectral': lambda WH: ELM_AE_FatSpectral_Pool2d(WH),
            'aggregationGAP' : lambda WH: aggregation_GAP(WH),
            # 'ELMspatiospectral' : lambda WH: SpatioSpectral_ELMs(WH),
            'ELMspatialEV': lambda WH: ELM_AE_Spatial_EVOLVED(WH),
            'ELMfatspectralEnsemble' : lambda WH : ELM_AE_FatSpectral_Ensemble(WH),
            'ELMaggregativeEnsemble': lambda WH: ELMaggregative(WH),
            'ELMaggregativeEnsembleM20': lambda WH: ELMaggregative(20)
            # 'Spectral_Conv' : lambda WH : Spectral_Conv(1)
            }


FEATURE_DEPTH = {'last': lambda d: -1,
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


class timm_pretrained_features(Module):    
    def __init__(self, model):
        super(timm_pretrained_features, self).__init__()        
        self.net = timm.create_model(model, features_only=True, pretrained=True)
        for param in self.net.parameters():
            param.requires_grad = False
        
        self.net.eval()
        
    def forward(self, x): #return all feature maps
        return self.net(x)   
    
    def get_features(self, x, depth='last', pooling='AvgPool2d', Q=1): 
        self.net.eval() #this is needed to disable batch norm, dropout, etc
        fmap = self.net(x)
        fmap = fmap[int(FEATURE_DEPTH[depth](len(fmap)))]   
        
        if 'ELM' in pooling:
            return torch.flatten(POOLINGS[pooling](Q)(fmap),
                                 start_dim=1, end_dim=-1)
        else:
            return torch.flatten(POOLINGS[pooling](fmap.size()[2])(fmap),
                                 start_dim=1, end_dim=-1)

### Inherits the internal functions of timm_pretrained_features for returning features  
#   the only difference here is that the model is created w/o pretrained weights  
class timm_random_features(timm_pretrained_features):    
    def __init__(self, model):
        super(timm_random_features, self).__init__(model)        
        self.net = timm.create_model(model, features_only=True, pretrained=False)
        for param in self.net.parameters():
            param.requires_grad = False   
        
        self.net.eval()
 
class timm_isotropic_features(Module):    
    def __init__(self, model, output_stride=8):
        super(timm_isotropic_features, self).__init__()  
        if output_stride==-1:
            self.net = timm.create_model(model, features_only=True, pretrained=True)
        else:
            try:
                self.net = timm.create_model(model, features_only=True, pretrained=True, output_stride=8)
            except:
                try:
                    self.net = timm.create_model(model, features_only=True, pretrained=True, output_stride=32)
                except:
                    self.net = timm.create_model(model, features_only=True, pretrained=True)
            
        for param in self.net.parameters():
            param.requires_grad = False   
        
        self.net.eval()
        
    def forward(self, x): #return all feature maps
        return self.net(x)  
    
    def get_features(self, x, depth='last', pooling='AvgPool2d', Q=1): 
        self.net.eval() #this is needed to disable batch norm, dropout, etc
        fmap = self.net(x)
        
        if 'ELM' in pooling: #only works with ELMfatspectral (aggregative)
            return torch.flatten(POOLINGS[pooling](Q)(fmap),
                             start_dim=1, end_dim=-1)
        else: #only works with aggregation GAP
            return torch.flatten(POOLINGS[pooling]()(fmap),
                                  start_dim=1, end_dim=-1)
        
        
class timm_attention_features(Module):    
    def __init__(self, model):
        super(timm_attention_features, self).__init__()        
        self.net = timm.create_model(model, num_classes=0, pretrained=True)
        for param in self.net.parameters():
            param.requires_grad = False
            
        self.net.eval()
        
    def forward(self, x): #return all feature maps
        return self.net(x)   
    
    def get_features(self, x, depth='last', pooling='AvgPool2d'): 
        self.net.eval()
        return self.net(x)

    
class ELM_AE_Spatial_EVOLVED(Module):
    def __init__(self, Q):
        super(ELM_AE_Spatial_EVOLVED, self).__init__()         
        self.Q = Q
        
    def forward(self, x):
        batch, z, w, h = x.size()
        Q=12
        N,P = z, w*h
        device=x.get_device()
        out = []
        # window_size = w
        # pos_encoding = ELM.positionalencoding2d(int(P), int(window_size), int(window_size))
        # pos_encoding = torch.reshape(pos_encoding, (P, N)).to(device)
        
        spatial_ELM = ELM.ELM_AE(Q=Q, P=P, N=N, device=device)
        func = Rearrange('b c h w -> b (h w) c') 
        x = func(x)
        for sample in x:
            sample = torch_zscore(sample, axis=0)
            sample = torch.nan_to_num(sample)
            spatial_features = spatial_ELM.fit_AE(sample)
            
            if Q==12:
                #flatten
                pooled_features=torch.reshape(spatial_features, (1, spatial_features.size()[0]*spatial_features.size()[1]))      
            else:
                pooled_features = torch.std(spatial_features, axis=1)
                
            out.append(pooled_features)
      
        return torch.stack(out)
   
class ELM_AE_Spatial_Pool2d(Module):
    def __init__(self, kernel_size):
        super(ELM_AE_Spatial_Pool2d, self).__init__() 
        
        self.wh = kernel_size
        
    def forward(self, x):
        batch, z, h, w = x.size()
        Q=1
        device=x.get_device()
        if h%2 != 0:
            SPATIAL_=h+1
        else:
            SPATIAL_=h
        out = []
        # errors = []
        window_size = int(np.round(0.25 * SPATIAL_))
        spatial_ELM = ELM.ELM_AE(Q=Q, P=(window_size**2), N=(SPATIAL_//window_size)**2, device='cpu')        
        patcher = Rearrange('b c (h p1) (w p2) -> b c (p1 p2) (h w)', p1 = window_size, p2 = window_size)
        
        x = torchvision.transforms.Resize(SPATIAL_)(x)
        x = patcher(x).cpu().detach().numpy()
        # print(x.size())
        for sample in x: 
            features_ = np.zeros((z))
            ii=0
            sample = stats.zscore(sample, axis=0)
            sample = np.nan_to_num(sample)
            for spectra in sample:            
                # feature_map = spectra.cpu().detach().numpy()
                # feature_map = self.get_spatial_matrix(feature_map, window_size,window_size)
                spatial_features = spatial_ELM.fit_AE(torch.from_numpy(spectra)).detach().numpy()
                
                # pooled_features =np.mean(np.abs(spatial_features), axis=0)
                pooled_features =np.mean(spatial_features, axis=0)                
                # pooled_features =np.std(spatial_features, axis=0)

                features_[ii] = pooled_features
                ii=ii+1
                
                # features_.append(pooled_features)
                # errors.append(erro)
                
            out.append(features_)
                  
        out = np.asarray(out, dtype=np.float64).reshape((batch, out[0].shape[0], 1))    
        return torch.from_numpy(out)
 

class ELM_AE_Spectral_Pool2d(Module):
    def __init__(self, Q):
        super(ELM_AE_Spectral_Pool2d, self).__init__()         
        self.Q = Q
        
    def forward(self, x):
        batch, z, w, h = x.size()
        Q=self.Q
        P,N = z, w*h
        device=x.get_device()
        out = []
        # window_size = w
        # pos_encoding = ELM.positionalencoding2d(int(P), int(window_size), int(window_size))
        # pos_encoding = torch.reshape(pos_encoding, (P, N)).to(device)
        
        spectral_ELM = ELM.ELM_AE(Q=Q, P=P, N=N, device=device)
        func = Rearrange('b c h w -> b c (h w)') 
        x = func(x)
        for sample in x:
            #doing positional encoding here
            # sample = torch.add(sample, pos_encoding)
            sample = torch_zscore(sample, axis=0)
            sample = torch.nan_to_num(sample)
            spectral_features = spectral_ELM.fit_AE(sample)
            
            if self.Q==1:
                #flatten
                pooled_features=torch.reshape(spectral_features, (1, spectral_features.size()[0]*spectral_features.size()[1]))      
            else:
                pooled_features = torch.std(spectral_features, axis=1)
                
            out.append(pooled_features)
      
        return torch.stack(out)

class ELMaggregative(Module):
    def __init__(self, Q):
        super(ELMaggregative, self).__init__()         
        self.Q = 1
        self.M = Q
        
    def forward(self, x): 
        SPATIAL_ = x[int(np.round(len(x)/2))].size()[2]
        batch, z, _, _ = x[0].size() #zero is the first activation map
        _, zn, _, _ = x[-1].size() #-1 is the last activation map
        Q=self.Q
        #ensemble size. M randomized AEs with different seed are summed after training
        M = self.M
        device = x[0].get_device()
        out = []
        P = sum([a.size()[1] for a in x])
        # pos_encoding = ELM.positionalencoding2d(int(P), SPATIAL_, SPATIAL_)
        # pos_encoding = ELM.torch.reshape(pos_encoding, (P, SPATIAL_**2)).to(x[0].get_device())
        
        ELMs = []
        for model in range(M):
            ELMs.append(ELM.ELM_AE(Q=Q, P=P, N=SPATIAL_**2, device=x[0].get_device(), seed=model*(Q*P)))
        # spectral_ELM2 = ELM.ELM_AE(Q=Q, P=P, N=SPATIAL_**2, device=x[0].get_device(), seed=9999)
        
        func = Rearrange('b c h w -> b c (h w)')        
        meta_feature_maps = torch.empty((batch, 0, SPATIAL_**2)).to(device)
             
        ######## layer-wise normalization of each sampe in the batch
        for depth in x:
            for sample in depth:
                ###euclidean norm over all axis of each layers output/activation map
                #spectral norm (dim=0) seems the best option
                torch.nn.functional.normalize(sample, p=2.0, dim=0, eps=1e-10, out=sample)
                
                ###zscore over all axis of each layers output/activation map
                # torch.divide(torch.subtract(sample, torch.mean(sample, dim=(0,1,2))), torch.std(sample, dim=(0,1,2)), out=sample)
                
        #stack activation maps accross z, for all layers used for feature extraction
        for ii in range(len(x)):            
            meta_feature_maps = torch.cat((meta_feature_maps, 
                                          func(torchvision.transforms.Resize(SPATIAL_)(x[ii]))), axis=1)
            x[ii]=[]
        x=[]        
        for sample in meta_feature_maps:
            # torch.nn.functional.normalize(sample, p=2.0, dim=0, eps=1e-10, out=sample)            
            # sample = torch_zscore(sample, axis=0)
            # sample= torch.nan_to_num(sample)
            
            # x_train = sample                   
            # spectral_features = spectral_ELM.fit_agg(x_train, target)
            spectral_features = torch.zeros(Q,P).to(device)
            for elm in ELMs:
                current_model=elm.fit_AE(sample)
                spectral_features = spectral_features + current_model

            if Q==1:
                pooled_features=torch.reshape(spectral_features, (1, spectral_features.size()[0]*spectral_features.size()[1]))
            else:
                pooled_features = torch.sum(spectral_features, axis=0)
              
            pooled_features= torch.nan_to_num(pooled_features)
            out.append(pooled_features)


        return torch.stack(out)

# class Spectral_Conv(Module):
#     def __init__(self, Q):
#         super(Spectral_Conv, self).__init__()         
#         self.Q = 16

#     def forward(self, x):
#         batch, z, w, h = x.size()
#         Q=self.Q
#         P,N = z, w*h
#         device=x.get_device()
#         out = []        
#         spectral_ELM = RATT.randomized_attention(Q=Q, P=P, N=N, device=device)
#         func = Rearrange('b c h w -> b c (h w)')  
#         x = func(x)
#         for sample in x:
#             for channel in sample:
#             # sample = torch_zscore(sample, axis=0)
#             # sample = torch.nan_to_num(sample)
#             # sample = torch.reshape(sample, (z, w, h))
#             # spectral_features = spectral_ELM.self_att(sample)    
#             pooled_features = [calc_entropy(i) for i in sample]
#             # if self.Q==1:
#             #     #flatten
#             #     pooled_features=torch.reshape(spectral_features, (1, spectral_features.size()[0]*spectral_features.size()[1]))
#             #     # pooled_features=spectral_features
#             # else:
#             #     pooled_features = torch.std(spectral_features, axis=0)
                
#             out.append(torch.stack(pooled_features))
      
#         return torch.stack(out)
    

class ELM_AE_FatSpectral_Pool2d(Module):
    def __init__(self, Q):
        super(ELM_AE_FatSpectral_Pool2d, self).__init__()         
        self.Q = Q
        
    def forward(self, x): 
        SPATIAL_ = x[int(np.round(len(x)/2))].size()[2]
        batch, _, _, _ = x[0].size() #zero is the last feature map
        Q=self.Q
        out = []
        P= sum([a.size()[1] for a in x])
        
        spectral_ELM = ELM.ELM_AE(Q=Q, P=P, N=SPATIAL_**2, device=x[0].get_device())
        func = Rearrange('b c h w -> b c (h w)')        
        
        meta_feature_maps = torch.empty((batch, 0, SPATIAL_**2)).to(x[0].get_device())
        for ii in range(len(x)):            
            meta_feature_maps = torch.cat((meta_feature_maps, 
                                          func(torchvision.transforms.Resize(SPATIAL_)(x[ii]))), axis=1)
            x[ii]=[]
        x=[]
        for sample in meta_feature_maps:
            sample = torch_zscore(sample, axis=0)
            sample= torch.nan_to_num(sample)
            spectral_features = spectral_ELM.fit_AE(sample)
            if Q==1:
                pooled_features=torch.reshape(spectral_features, (1, spectral_features.size()[0]*spectral_features.size()[1]))
            else:
                pooled_features = torch.std(spectral_features, axis=1)
                
            out.append(pooled_features)
            sample=[]

        return torch.stack(out)

class ELM_AE_FatSpectral_Ensemble(Module):
    def __init__(self, Q):
        super(ELM_AE_FatSpectral_Ensemble, self).__init__()         
        self.Q = Q
        
    def forward(self, x): 
        batch, _, _, _ = x[0].size() #zero is the last feature map
        SPATIAL_ = x[int(np.round(len(x)/2))].size()[2]
        # FEATURE_DEPTH['middle'](len(x))
        Q=self.Q
        out = []  
        device = x[0].get_device()
        func = Rearrange('b c h w -> b c (h w)') 
        ELMs = []
        final_z = 0
        for ii in range(len(x)):   
            x[ii] =  func(torchvision.transforms.Resize(SPATIAL_)(x[ii]))
            _, z, wh = x[ii].size()
            final_z+=z
            ELMs.append(ELM.ELM_AE(Q=Q, P=z, N=wh, device=device))
                    
        for sample in range(batch):
            sample_features = torch.zeros((final_z)).to(device)
            z_index = 0
            for ii in range(len(x)):                 
                spectral_features = ELMs[ii].fit_AE(torch.nan_to_num(torch_zscore(x[ii][sample], axis=0)))
                if Q==1:
                    pooled_features=torch.reshape(spectral_features, (1, spectral_features.size()[0]*spectral_features.size()[1]))
                    sample_features[z_index:z_index + pooled_features.size()[1]] = pooled_features
                else:
                    pooled_features = torch.var(spectral_features, axis=1)
                    sample_features[z_index:z_index + pooled_features.size()[0]] = pooled_features              
                    
                z_index += pooled_features.size()[0]
                # sample_features = torch.cat((sample_features, pooled_features))
                                
            out.append(sample_features)                

        return torch.stack(out)


class aggregation_GAP(Module):
    def __init__(self):
        super(aggregation_GAP, self).__init__()        
    
    def forward(self, x):
        batch, z, w, h = x[0].size()
              
        meta_feature_maps = torch.empty((batch, 0)).to(x[0].get_device())
        for ii in range(len(x)): 
            meta_feature_maps = torch.cat((meta_feature_maps,
                                           torch.flatten(nn.AvgPool2d(x[ii].size()[2])(x[ii]),
                                             start_dim=1, end_dim=-1)), axis=1)
            
        return meta_feature_maps    
  
def torch_zscore(x, axis):
    avgs = torch.mean(x, axis=axis)
    stds = torch.std(x, axis=axis)
    if axis==0:
        return (x-avgs)/stds
    else:
        return ((x.transpose(1,0)-avgs)/stds).transpose(1,0)






    