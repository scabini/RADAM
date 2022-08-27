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
from code_RNN import ELM
from einops.layers.torch import Rearrange


POOLINGS = {'AvgPool2d': lambda WH: nn.AvgPool2d(WH),
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
            'ELMspatiospectral' : lambda WH: SpatioSpectral_ELMs(WH),
            'ELMspatiotropic': lambda WH: ELM_AE_Spatial_EVOLVED(WH)
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
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class timm_pretrained_features(Module):    
    def __init__(self, model):
        super(timm_pretrained_features, self).__init__()        
        self.net = timm.create_model(model, features_only=True, pretrained=True)
        for param in self.net.parameters():
            param.requires_grad = False
            
        self.net.eval()
        
    def forward(self, x): #return all feature maps
        return self.net(x)   
    
    def get_features(self, x, depth='last', pooling='AvgPool2d'): 
        self.net.eval() #this is needed to disable batch norm, dropout, etc
        fmap = self.net(x)
        fmap = fmap[int(FEATURE_DEPTH[depth](len(fmap)))]   
        
        # io.savemat('feature_maps.mat', {'all': fmap.cpu().detach().numpy()})
        
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
                self.net = timm.create_model(model, features_only=True, pretrained=True, output_stride=32)
            
        for param in self.net.parameters():
            param.requires_grad = False   
        
        self.net.eval()
        
    def forward(self, x): #return all feature maps
        return self.net(x)  
    
    def get_features(self, x, depth='last', pooling='AvgPool2d'): 
        self.net.eval() #this is needed to disable batch norm, dropout, etc
        fmap = self.net(x)
        
        return torch.flatten(POOLINGS[pooling](len(fmap))(fmap),
                             start_dim=1, end_dim=-1)
    
class ELM_AE_Spatial_EVOLVED(Module):
    def __init__(self, kernel_size):
        super(ELM_AE_Spatial_EVOLVED, self).__init__() 
        
        self.wh = kernel_size
        
    def forward(self, x):
        batch, z, h, w = x[-1].size()
        Q=1
        SPATIAL_ = x[int(np.round(len(x)/2))].size()[2]
        x = x[-1]
        out = []
        window_size = 4
        
        # patcher = Rear'range('b c (h p1) (w p2) -> b c (p1 p2) (h w)', p1 = window_size, p2 = window_size)
        
        x = torchvision.transforms.Resize(SPATIAL_)(x)
        
        batch_size, n_channels, n_rows, n_cols = batch, z, h, w
        kernel_h, kernel_w = window_size, window_size
        step = 2
                
        # unfold(dimension, size, step)
        x = x.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(2, 3, 0, 1, 4, 5).reshape(batch_size, n_channels, -1, kernel_h, kernel_w)
        
        spatial_ELM = RNN.RNN_AE(Q=Q, P=(window_size**2), N=x.size()[2])        
        # print(x.size(), batch, z, h, SPATIAL_)
        # x = patcher(x)
        # print(((window_size**2), Q, (window_size**2)), (SPATIAL_//window_size)**2)
        window_size = window_size**2
        for sample in x: 
            features_ = np.zeros((z))
            ii=0
            sample = sample.cpu().detach().numpy()
            
            sample = stats.zscore(sample, axis=0)
            sample = np.nan_to_num(sample)
            for spectra in sample:          
                spectra = spectra.reshape((x.size()[2], window_size)).transpose()
                spatial_features, erro = spatial_ELM.fit(spectra, axis=0)
                
                pooled_features =np.mean(spatial_features, axis=0)   
                
                features_[ii] = pooled_features
                ii=ii+1
   
            out.append(features_)
                  
        out = np.asarray(out, dtype=np.float64).reshape((batch, out[0].shape[0], 1))
        # print(np.mean(errors), np.std(errors))
        # print('Q=', Q, 'features=', (np.min(out), np.max(out)), )#'ERRO:', np.mean(errors))

        return torch.from_numpy(out)


class ELM_AE_Spatial_Pool2d(Module):
    def __init__(self, kernel_size):
        super(ELM_AE_Spatial_Pool2d, self).__init__() 
        
        self.wh = kernel_size
        
    def forward(self, x):
        batch, z, h, w = x.size()
        Q=1
        SPATIAL_=8
        out = []
        # errors = []
        window_size = 2
        spatial_ELM = RNN.RNN_AE(Q=Q, P=(window_size**2), N=(SPATIAL_//window_size)**2)        
        patcher = Rearrange('b c (h p1) (w p2) -> b c (p1 p2) (h w)', p1 = window_size, p2 = window_size)
        
        x = torchvision.transforms.Resize(SPATIAL_)(x)
        x = patcher(x)
        # print(x.size())
        for sample in x: 
            features_ = np.zeros((z))
            ii=0
            sample = sample.cpu().detach().numpy()
            sample = stats.zscore(sample, axis=0)
            sample = np.nan_to_num(sample)
            for spectra in sample:            
                # feature_map = spectra.cpu().detach().numpy()
                # feature_map = self.get_spatial_matrix(feature_map, window_size,window_size)
                spatial_features, erro = spatial_ELM.fit(spectra, axis=0)
                
                # pooled_features =np.mean(np.abs(spatial_features), axis=0)
                pooled_features =np.mean(spatial_features, axis=0)                
                # pooled_features =np.std(spatial_features, axis=0)
                
                features_[ii] = pooled_features
                ii=ii+1
                # pooled_features =np.hstack((spatial_features,spectral_features))    
                # pooled_features=np.reshape(spatial_features, (spatial_features.shape[0]*spatial_features.shape[1]))
                # features_.append(pooled_features)
                # errors.append(erro)
                
            out.append(features_)
                  
        out = np.asarray(out, dtype=np.float64).reshape((batch, out[0].shape[0], 1))
        # print(np.mean(errors), np.std(errors))
        # print('Q=', Q, 'features=', (np.min(out), np.max(out)), )#'ERRO:', np.mean(errors))

        return torch.from_numpy(out)
    

class ELM_AE_Spectral_Pool2d(Module): #numpy version
    def __init__(self, kernel_size):
        super(ELM_AE_Spectral_Pool2d, self).__init__()         
        self.wh = kernel_size
        
    def forward(self, x):
        batch, z, w, h = x.size()
        Q=1
        out = []
        errors = []
        
        spectral_ELM = RNN.RNN_AE(Q=Q, P=z, N=w*h)
        # spatial_ELM = RNN.RNN_AE(Q=512, P=w*h, N=z)
        
        x = x.cpu().detach().numpy()
        for sample in x:            
            feature_map = np.reshape(sample, (z, w*h))
            feature_map = stats.zscore(feature_map, axis=0)
            feature_map = np.nan_to_num(feature_map)
            spectral_features, erro = spectral_ELM.fit(feature_map, axis=0)
            # pooled_features =np.mean(np.abs(spectral_features), axis=1)
            pooled_features=np.reshape(spectral_features, (spectral_features.shape[0]*spectral_features.shape[1]))
            errors.append(erro)
            out.append(pooled_features)
                  
        out = np.asarray(out, dtype=np.float64).reshape((batch, pooled_features.shape[0], 1))
        # print(np.mean(errors), np.std(errors))
        # print('Q=', Q, 'W=', (np.min(out), np.max(out)), 'ERRO:', np.mean(errors))

        return torch.from_numpy(out)

    
class ELM_AE_FatSpectral_Pool2d(Module):
    def __init__(self, kernel_size):
        super(ELM_AE_FatSpectral_Pool2d, self).__init__() 
        
        self.feature_blocks = kernel_size
        
    def forward(self, x):        
        SPATIAL_ = x[int(np.round(len(x)/2))].size()[2]
        batch, _, _, _ = x[0].size() #zero is the last feature map
        Q=1
        out = []
        errors = []
        P= sum([a.size()[1] for a in x])
        
        spectral_ELM = RNN.RNN_AE(Q=Q, P=P, N=SPATIAL_**2)
        for sample in range(batch):
            meta_feature_map = np.empty((0,SPATIAL_**2), dtype=np.float64)
            for depth in x:
                z, w, h = depth[sample].size()
                feature_map = torchvision.transforms.Resize(SPATIAL_)(depth[sample])
                
                feature_map = torch.reshape(feature_map, (z, SPATIAL_**2))
                # feature_map = torch.div(feature_map, torch.max(feature_map))
                feature_map = feature_map.cpu().detach().numpy()
                meta_feature_map = np.vstack((meta_feature_map, feature_map))
             
            # pooled_features,erro = RNN.RNN_AE(meta_feature_map,
            #                       Q=Q)
            
            meta_feature_map = stats.zscore(meta_feature_map, axis=0)
            meta_feature_map = np.nan_to_num(meta_feature_map)
            
            pooled_features,erro = spectral_ELM.fit(meta_feature_map, axis=None)
            pooled_features=np.reshape(pooled_features, (pooled_features.shape[0]*pooled_features.shape[1]))
            
            errors.append(erro)
            out.append(pooled_features)
        
        out = np.asarray(out, dtype=np.float64).reshape((batch, pooled_features.shape[0], 1))
        # print(SPATIAL_, Q, P, SPATIAL_**2, meta_feature_map.shape)
        # print(SPATIAL_, meta_feature_map.shape, np.min(out), np.max(out))
        # print('Q=', Q, SPATIAL_, 'W=', (np.min(out), np.max(out)), 'ERRO:', np.mean(errors))
        return torch.from_numpy(out)    


class SpatioSpectral_ELMs(Module): #numpy version
    def __init__(self, kernel_size):
        super(SpatioSpectral_ELMs, self).__init__()         
        self.wh = kernel_size
        
    def forward(self, x):        
        batch, z, h, w = x.size()
        Q=1
        SPATIAL_=8        
        window_size = 2      
        spectral_ELM = RNN.RNN_AE(Q=Q, P=z, N=SPATIAL_**2) 
        spatial_ELM = RNN.RNN_AE(Q=Q, P=(window_size**2), N=(SPATIAL_//window_size)**2)        
        patcher = Rearrange('b c (h p1) (w p2) -> b c (p1 p2) (h w)', p1 = window_size, p2 = window_size)
        
        out_spat = []
        out = []  
        x = torchvision.transforms.Resize(SPATIAL_)(x)
        x = patcher(x)
        
        for sample in x: 
            
            sample = sample.cpu().detach().numpy()
            sample = stats.zscore(sample, axis=0)
            sample = np.nan_to_num(sample)
            
            feature_map = np.reshape(sample, (z, SPATIAL_**2))            
            spectral_features, _ = spectral_ELM.fit(feature_map, axis=0)            
            spectral_features=np.reshape(spectral_features, (spectral_features.shape[0]*spectral_features.shape[1]))           
            out.append(spectral_features)
            
            features_ = np.zeros((z))
            ii=0
            
            for spectra in sample:    
                spatial_features, _ = spatial_ELM.fit(spectra, axis=0)
                pooled_features =np.mean(spatial_features, axis=0)  
                
                features_[ii] = pooled_features
                ii=ii+1
                
            out_spat.append(features_)        
        
        out_spat = np.asarray(out_spat, dtype=np.float64).reshape((batch, out_spat[0].shape[0], 1))
        out = np.asarray(out, dtype=np.float64).reshape((batch, out[0].shape[0], 1))


        return torch.from_numpy(np.concatenate((out_spat, out), axis=1,dtype=np.float64))

    
class aggregation_GAP(Module):
    def __init__(self, kernel_size):
        super(aggregation_GAP, self).__init__() 
        
        self.feature_blocks = kernel_size
        
    def forward(self, x):
        batch, z, w, h = x[0].size()
        out =[]
        for sample in range(batch):
            meta_feature_map = np.empty((0), dtype=np.float64)
            for depth in x:
                z, w, h = depth[sample].size()
                
                feature_map = torch.reshape(depth[sample], (z, w*h))
                # feature_map = torch.div(feature_map, torch.max(feature_map))
                feature_map = feature_map.cpu().detach().numpy()
                meta_feature_map = np.hstack((meta_feature_map, np.mean(feature_map, axis=1)))
           
            out.append(meta_feature_map)   
        out = np.asarray(out, dtype=np.float64).reshape((batch, out[0].shape[0], 1))
        return torch.from_numpy(out)     
    