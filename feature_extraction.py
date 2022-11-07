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
import models
# import extractor_models

###method is a string, the model name like in timm. If you put '-random' in it,
#   tries to download pretrained weights for 'backbone' and extract features using 'pooling' technique
#   tries to use cuda for that, use CPU if no cuda device is available
#   seed is used to improve reproducibility
#   M only works for the ELM methods, otherwise it will just be ignored
def extract_features(backbone, dataset, pooling, seed, depth='all', multigpu=False, batch_size=1, M=1):
    
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
        
    # gpus = torch.cuda.device_count()
    total_cores=multiprocessing.cpu_count()
    num_workers = total_cores

    # print("System has", gpus, "GPUs and", total_cores, "CPU cores. Using", num_workers, "cores for data loaders.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = dataset[0][0].size()[2]
    
    ### Creating the model
    if 'RAE' in pooling and depth == 'all': #aggregative RAE use output_stride to control spatial dimms
        model = models.timm_feature_extractor(backbone, output_stride=True, input_dim=input_dim,
                                              depth=depth, pooling=pooling, M=M, device=device)
    else: 
        model = models.timm_feature_extractor(backbone, output_stride=False, input_dim=input_dim,
                                              depth=depth, pooling=pooling, M=M, device=device)    
      
    model.net.to(device)
    
    data_loader = torch.utils.data.DataLoader(dataset,
                          batch_size=batch_size, shuffle = False, drop_last=False, pin_memory=True, num_workers=num_workers)  
   
    feature_size = model(next(iter(data_loader))[0].to(device)).cpu().detach().numpy().shape[1]
    if feature_size > 10000:
        return 'error' #ignoring backbones with huge number of features
    # model()     
    print('extracting', feature_size, 'features for', len(dataset), 'images...')
    X = np.empty((0,feature_size))
    Y = np.empty((0))
    
    
    if multigpu:
        model.net = nn.DataParallel(model.net)
        
    for i, data in enumerate(data_loader, 0):
      
        inputs, labels = data[0].to(device), data[1]   
              
        X = np.vstack((X,model(inputs).cpu().detach().numpy()))

        Y = np.hstack((Y, labels))

    del model.net
    del model
    del data_loader
    del dataset
    return X,Y

# def extract_features_custom_nodes(method, dataset, pooling, seed, depth='last', multigpu=False, batch_size=1, Q=1):
    
#     torch.manual_seed(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.use_deterministic_algorithms(True)
        
#     # gpus = torch.cuda.device_count()
#     total_cores=multiprocessing.cpu_count()
#     num_workers = total_cores

#     # print("System has", gpus, "GPUs and", total_cores, "CPU cores. Using", num_workers, "cores for data loaders.")
#     device = "cuda" if torch.cuda.is_available() else "cpu"
           
#     if '-random' in method: #random features?
#         model = extractor_models.extractor_random_features(method.split('-')[0]) 
#     elif '-aggregation' in method:
#         model = extractor_models.extractor_isotropic_features(method.split('-')[0])
#     else: #else: try to download/create the pretrained version
#         model = extractor_models.extractor_pretrained_features(method)    
      
#     model.net.to(device)
    
#     data_loader = torch.utils.data.DataLoader(dataset,
#                           batch_size=batch_size, shuffle = False, drop_last=False, pin_memory=True, num_workers=num_workers)  
   
#     feature_size = model.get_features(next(iter(data_loader))[0].to(device), depth, pooling).cpu().detach().numpy().shape[1]
#     # model()     
#     # print('extracting', feature_size, 'features...')
#     X = np.empty((0,feature_size))
#     Y = np.empty((0))
    
    
#     if multigpu:
#         model.net = nn.DataParallel(model.net)
        
#     for i, data in enumerate(data_loader, 0):
      
#         inputs, labels = data[0].to(device), data[1]   
              
#         X = np.vstack((X,model.get_features(inputs, depth, pooling, Q=Q).cpu().detach().numpy()))

#         Y = np.hstack((Y, labels))

#     del model.net
#     del model
#     del data_loader
#     return X,Y
    
        
        
            
            
            
            