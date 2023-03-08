# -*- coding: utf-8 -*-
"""
This is the PyTorch implementation of the RADAM module, along with a simple example.
By changing the "model" variable, you may choose different backbones to be coupled
with our method, and it works with almost any timm model with the 'features_only' option!
    
To effectively train RADAM, you need to attach a linear classifier at the top
(for the paper we used a linear SVM from sklearn). You may then use one of the 
datasets we explored in the paper since they are public. Eg.:
    `Describable Textures Dataset (DTD) <https://www.robots.ox.ac.uk/~vgg/data/dtd/>`_.
    https://pytorch.org/vision/main/generated/torchvision.datasets.DTD.html
    
You can also find a 'requirements.yml' file attached, with the libraries we used.
"""

model = 'convnext_nano' #select the timm backbone to be coupled with RADAM

import timm
net = timm.create_model(model, features_only=True, pretrained=True, output_stride=8)

import numpy as np
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
net.to(device)
act_maps = net(torch.zeros(1,3,224,224))
z = sum([d.size()[1] for d in act_maps])
half_depth = int(np.round(len(act_maps)/2))
w = act_maps[half_depth].size()[2]
h = act_maps[half_depth].size()[3]

import torchvision
from einops.layers.torch import Rearrange

class RADAM(torch.nn.Module):
    def __init__(self, device, z, spatial_dims = (28,28), m=4, pos_encoding=True, dim_norm=(2,3)):
        super(RADAM, self).__init__()     
        self.q=1 
        self.z=z
        self.device = device
        self.rearange = torch.nn.Sequential(lp_norm_layer(p=2.0, dim=dim_norm, eps=1e-10),
                                      torchvision.transforms.Resize(spatial_dims),
                                      Rearrange('b c h w -> b c (h w)')
                                     )        
        self.RAEs = []
        for model in range(m):
            self.RAEs.append(RAE(q=self.q, z=z, w=spatial_dims[0], h=spatial_dims[1], device=device, seed=model*(self.q*z), pos_encoding=pos_encoding))
     
    def forward(self, x):        
        out = []
        if not isinstance(x, list): 
            x = [x]             
        device = self.device
        batch, _, _, _ = x[0].size()
        for depth in range(len(x)): 
            x[depth] = self.rearange(x[depth])             
        for sample in range(batch):            
            agg_activation_map = torch.vstack([x[i][sample] for i in range(len(x))])      
            pooled_features = torch.zeros(self.q,self.z).to(device)            
            for rae in self.RAEs:
                pooled_features += rae.fit_AE(agg_activation_map)
            out.append(pooled_features)
        return torch.stack(out)

class lp_norm_layer(torch.nn.Module):
    def __init__(self,p=2.0, dim=(1,2), eps=1e-10):
        super().__init__() 
        self.p=p
        self.dim=dim
        self.eps=eps
    def forward(self, x):
        return torch.nn.functional.normalize(x, p=self.p, dim=self.dim, eps=self.eps)
    
class RAE():
    def __init__(self, q, z, w, h, device, pos_encoding, seed):
        self._device = device
        self.pos_encoding=pos_encoding    
        self.W = make_orthogonal(LCG(q, z, seed)).to(device)
        if self.pos_encoding:
            self.encoding = positionalencoding2d(z, w, h)
            self.encoding = torch.reshape(self.encoding, (z, w*h)).to(device)
        self._activation = torch.sigmoid
     
    def fit_AE(self, x):     
        if self.pos_encoding:
            x = torch.add(x, self.encoding)
        g = self._activation(torch.mm(self.W, x))
        return torch.linalg.lstsq(g.t(), x.t()).solution

import pickle
def LCG(m, n, seed):
    L = m*n    
    # To generate this weight file, use:    
        #V = torch.zeros(2**18, dtype=torch.float)    
        #V[0], a, b, c = 1, 75, 74, (2**16)+1         
        #for x in range(1, (2**18)):
        #   V[x] = (a*V[x-1]+b) % c
        #with open('RAE_LCG_weights.pkl', 'wb') as f:
            #pickle.dump(V, f)
    with open('RAE_LCG_weights.pkl', 'rb') as f:
        V = pickle.load(f)
        f.close()      
    V = V[seed:L+seed] 
    V = torch.divide(torch.subtract(V, torch.mean(V)), torch.std(V))
    return V.reshape((m,n))
               
def make_orthogonal(tensor):
    #similar to torch.nn.init.orthogonal_
    rows = tensor.size(0)
    cols = tensor.numel() // rows
    flattened = torch.reshape(tensor, (rows, cols))
    if rows < cols:
        flattened.t_()
    q, r = torch.linalg.qr(flattened)
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph
    if rows < cols:
        q.t_()        
    return q

import math
def positionalencoding2d(d_model, height, width):
    pe = torch.zeros(d_model, height, width)
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    return pe

#testing with an empty input tensor
input_batch = torch.zeros(1,3,224,224)
texture_representation = RADAM(device, z, (w,h))(net(input_batch))

#You can then use 'texture_representation's of a set of images to train a
#   linear SVM, eg:
# from sklearn import svm
# SVM = svm.SVC(kernel='linear')
