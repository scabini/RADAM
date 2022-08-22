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

POOLINGS = {'AvgPool2d': lambda WH: nn.AvgPool2d(WH),
            'MaxPool2d': lambda WH: nn.MaxPool2d(WH),
            # 'AdaptiveAvgPool2d': lambda WH: nn.AdaptiveAvgPool2d(int(1)), #same as standard pooling when doing global pooling
            # 'AdaptiveMaxPool2d': lambda WH: nn.AdaptiveMaxPool2d(int(1)),
            'FractionalMaxPool2d': lambda WH: nn.FractionalMaxPool2d(WH, output_size=1),
            'LPPool2d': lambda WH: nn.LPPool2d(2, WH),
            'ELM_spatial': lambda WH: ELM_AE_SpatialAutomatic_Pool2d(WH),
            'ELM_spectral': lambda WH: ELM_AE_Spectral_Pool2d(WH),
            'ELM_spectral_torch': lambda WH: ELM_AE_Spectral_Pool2d_torch(WH),
            'ELM_fatspectral': lambda WH: ELM_AE_FatSpectral_Pool2d(WH)
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
    def __init__(self, model):
        super(timm_isotropic_features, self).__init__()        
        self.net = timm.create_model(model, features_only=True, pretrained=True, output_stride=8)
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
    

 
class ELM_AE_SpatialAutomatic_Pool2d(Module):
    def __init__(self, kernel_size):
        super(ELM_AE_SpatialAutomatic_Pool2d, self).__init__() 
        
        self.wh = kernel_size
        
    def forward(self, x):
        batch, z, w, h = x.size()
        out = []
        for sample in x:            
            feature_map = torch.reshape(sample, (z, w*h))
            feature_map = torch.div(feature_map, torch.max(torch.abs(feature_map)))
            feature_map = torch.add(feature_map, 1)
            
            feature_map = np.float64(feature_map.cpu().detach().numpy())
            
            feature_map = [[stats.entropy(data, base=2)] for data in feature_map]
            pooled_features = np.asarray(feature_map).transpose()
            
            # pooled_features = RNN.RNN_AE(feature_map,
            #                       Q=z//(4))
            
            pooled_features=np.reshape(pooled_features, (pooled_features.shape[0]*pooled_features.shape[1]))
            
            out.append(pooled_features)
                  
        out = np.asarray(out, dtype=np.float64).reshape((batch, len(pooled_features), 1))
        return torch.from_numpy(out)

    
class ELM_AE_Spectral_Pool2d_torch(Module): #NEW TORCH VERSION
    def __init__(self, kernel_size):
        super(ELM_AE_Spectral_Pool2d_torch, self).__init__() 
        
        self.wh = kernel_size
        
    def forward(self, x):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch, z, w, h = x.size()
        Q=1
        out = torch.empty(0, Q*z).to(device)
        # errors = []
        for sample in x:            
            feature_map = torch.reshape(sample, (z, w*h))
            # feature_map = torch.div(feature_map, torch.max(feature_map))
            # feature_map = feature_map.cpu().detach().numpy()
            # feature_map = stats.zscore(feature_map, axis=1)
            # feature_map = torch.divide(torch.subtract(feature_map, torch.mean(feature_map)), torch.var(feature_map))
            # X = np.nan_to_num(X)
            
            pooler = ELM.ELM(z, Q, device=device)
            pooler.fit(feature_map)

            pooled_features = pooler.get_output_weights()
            # pooled_features = torch.nan_to_num(pooled_features)
            
            # errors.append(erro)
            # pooled_features=np.reshape(pooled_features, (pooled_features.shape[0]*pooled_features.shape[1]))
            
            # pooled_features = np.hstack((pooled_features, np.mean(feature_map, axis=1)))
            
            # out.append(pooled_features)
            out = torch.vstack((out, pooled_features))
                  
        # out = torch.asarray(out, dtype=torch.float64).reshape((batch, Q*z, 1))
        # print(np.mean(errors), np.std(errors))
        return out.reshape((batch, Q*z, 1))

class ELM_AE_Spectral_Pool2d(Module): #numpy version
    def __init__(self, kernel_size):
        super(ELM_AE_Spectral_Pool2d, self).__init__() 
        
        self.wh = kernel_size
        
    def forward(self, x):
        batch, z, w, h = x.size()
        Q=1
        out = []
        errors = []
        for sample in x:            
            feature_map = torch.reshape(sample, (z, w*h))
            # feature_map = torch.div(feature_map, torch.max(feature_map))
            feature_map = feature_map.cpu().detach().numpy()

            pooled_features, erro = RNN.RNN_AE(feature_map,
                                  Q=Q)
            errors.append(erro)
            # pooled_features = np.nan_to_num(pooled_features)
            pooled_features=np.reshape(pooled_features, (pooled_features.shape[0]*pooled_features.shape[1]))
            
            # pooled_features = np.hstack((pooled_features, np.mean(feature_map, axis=1)))
            
            out.append(pooled_features)
                  
        out = np.asarray(out, dtype=np.float64).reshape((batch, Q*z, 1))
        # print(np.mean(errors), np.std(errors))
        print('Q=', Q, feature_map.shape, np.min(out), np.max(out))

        return torch.from_numpy(out)

    
class ELM_AE_FatSpectral_Pool2d(Module):
    def __init__(self, kernel_size):
        super(ELM_AE_FatSpectral_Pool2d, self).__init__() 
        
        self.feature_blocks = kernel_size
        
    def forward(self, x):
        
        # SPATIAL_ =  x[2].size()[2]
        SPATIAL_ =  8
        batch, z, w, h = x[0].size()
        # Q= [a.size()[1] for a in x]
        # Q = sum(Q)//x[-1].size()[1]
        Q=1
        # Q= int(np.round(x[-1].size()[1] / (SPATIAL_**2))) if int(np.round(x[-1].size()[1] / (SPATIAL_**2)))>0 else 1
        # print(Q)
        out = []
        for sample in range(batch):
            meta_feature_map = np.empty((0,SPATIAL_**2), dtype=np.float64)
            for depth in x:
                z, w, h = depth[sample].size()
                feature_map = torchvision.transforms.Resize(SPATIAL_)(depth[sample])
                
                feature_map = torch.reshape(feature_map, (z, SPATIAL_**2))
                # feature_map = torch.div(feature_map, torch.max(feature_map))
                feature_map = feature_map.cpu().detach().numpy()
                meta_feature_map = np.vstack((meta_feature_map, feature_map))
             
            pooled_features,erro = RNN.RNN_AE(meta_feature_map,
                                  Q=Q)
            
            pooled_features=np.reshape(pooled_features, (pooled_features.shape[0]*pooled_features.shape[1]))
            
            out.append(pooled_features)
        
        out = np.asarray(out, dtype=np.float64).reshape((batch, Q*meta_feature_map.shape[0], 1))
        print(SPATIAL_, meta_feature_map.shape, np.min(out), np.max(out))
        return torch.from_numpy(out)    
    
    
    