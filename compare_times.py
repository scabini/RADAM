#%%
import torch
from torch.utils.data import DataLoader
from feature_extraction import *
from models import *
from train import *
import datasets
import os
import pickle
import timm
from torch import optim
from torch.utils import data
from torchvision import datasets as torchDatasets
from torchvision import transforms
import numpy as np
import times

#%%
averages =  (0.485, 0.456, 0.406)
variances = (0.229, 0.224, 0.225)  
_transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(averages, variances),
        ])  

#%%

backbones = ['mobilenetv2_100','mobilenetv2_140', 'resnet18', 'resnet50', 'convnext_nano', 'convnext_tiny', 'convnext_base', 'convnext_large', 'convnext_xlarge_in22k']
devices = []
if torch.cuda.is_available(): devices.append("cuda:1") 
devices.append("cpu")
for device in devices:
    print(f"---------------------------------{device}----------------------------------------------")
    for backbone in backbones:
        batch_size=1
        arr_lab = np.ones(batch_size)
        model = timm.create_model(backbone, pretrained=True, num_classes=100).to(device)
        input_tensor = _transform(torch.rand(batch_size,3,224,224).to(device))
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        train_dataloader = DataLoader(input_tensor, batch_size=batch_size, shuffle=False, num_workers=0)
        times_forward = []
        times_backward = []
        times_RAE = []

        epochs=100
        for epoch in range(epochs):
            
            #Train
            train_loss, train_acc = train(train_dataloader, model, optimizer, criterion, epoch, device)

        times_backward = times.times_backward
        times_forward = times.times_forward


        model = models.timm_feature_extractor(backbone, output_stride=True, input_dim=224,
                                                    depth='all', pooling='RAEspatial', M=1, device=device)
        model.net.to(device)
        for _ in range(100):
            model(input_tensor)

        times_RAE = times.times_RAE
          
        times_forward = np.array(times_forward)
        times_backward = np.array(times_backward)
        times_RAE = np.array(times_RAE)

        print(f"{backbone} & {np.mean(times_forward)*1000:.3f}\\pm{{{np.std(times_forward):.2f}}} & {np.mean(times_backward)*1000:.3f}\\pm{{{np.std(times_backward):.2f}}} & {np.mean(times_RAE)*1000:.3f}\\pm{{{np.std(times_RAE):.2f}}}")
