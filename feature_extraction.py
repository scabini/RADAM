# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 14:40:21 2022

Feature extraction with timm models

@author: scabini
"""

import multiprocessing
import numpy as np
import random
import torch
import torch.nn as nn
from torchvision import transforms
import models

###method is a string, the model name like in timm. If you put '-random' in it,
#   random weights are used for feature extraction. Otherwise, tries to
#   download pretrained weights
#   the random seed is only for using random models
def extract_features(method, dataset, pooling, input_dimm=224, depth='last', multigpu=False, batch_size=1, seed=666999):
    
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
        
    # gpus = torch.cuda.device_count()
    total_cores=multiprocessing.cpu_count()
    num_workers = total_cores // 2

    # print("System has", gpus, "GPUs and", total_cores, "CPU cores. Using", num_workers, "cores for data loaders.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
        
    ### Creating the model
    if '-random' in method: #random features?
        model = models.timm_random_features(method.split('-')[0])  
    elif '-aggregation' in method:
        model = models.timm_isotropic_features(method.split('-')[0])
    else: #else: try to download/create the pretrained version
        model = models.timm_pretrained_features(method)    
      
    model.net.to(device)
    
    # ImageNet normalization parameters
    averages =  (0.485, 0.456, 0.406)
    variances = (0.229, 0.224, 0.225)
    input_dimm = input_dimm    
    #Data loader
    _transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(averages, variances),
        transforms.Resize((input_dimm,input_dimm))
        # transforms.CenterCrop(128)
    ])    
    
    dataset= dataset(_transform) 

    data_loader = torch.utils.data.DataLoader(dataset,
                          batch_size=batch_size, shuffle = False, drop_last=False, pin_memory=True, num_workers=num_workers)  
   
    feature_size = model.get_features(torch.zeros((1,3, input_dimm,input_dimm)).to(device), depth, pooling).cpu().detach().numpy().shape[1]
         
    print('extracting', feature_size, 'features...')
    X = np.empty((0,feature_size))
    Y = np.empty((0))
    
    
    if multigpu:
        model.net = nn.DataParallel(model.net)
        
    for i, data in enumerate(data_loader, 0):
      
        inputs, labels = data[0].to(device), data[1]   
              
        X = np.vstack((X,model.get_features(inputs, depth, pooling).cpu().detach().numpy()))

        Y = np.hstack((Y, labels))

    del model
    return X,Y
    
        
        
            
            
            
            