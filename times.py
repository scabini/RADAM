from TESTS_finalRADAM import RADAM
import timm
import numpy as np
import torch
import time
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"]='0'
os.environ["OMP_NUM_THREADS"]='1'

model = 'convnext_nano' #select the timm backbone to be coupled with RADAM

for device in ['cpu', 'cuda']:
    net = timm.create_model(model, features_only=True, pretrained=False, output_stride=8)
    net.eval()
    # device = "cuda"
    # device = "cpu"
    net.to(device)
    act_maps = net(torch.zeros(1,3,224,224).to(device))
    z = sum([d.size()[1] for d in act_maps])
    half_depth = int(np.round(len(act_maps)/2))
    w = act_maps[half_depth].size()[2]
    h = act_maps[half_depth].size()[3]
    
    forward_time = []
    backward_time = []
    RADAM_time = []
    texture_encoder = RADAM(device, z, (w,h), m=4)
    criterion = torch.nn.CrossEntropyLoss()
    
    
    
    for i in range(100):
        seed = i
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        net_temp = timm.create_model(model, pretrained=False, num_classes=10)
        net_temp.to(device)
        net_temp.train()
        
        input = torch.rand(1,3,224,224).to(device) - 0.5
        
        start_time = time.time()
        output = net_temp(input)
        forward_time.append(time.time() - start_time)
    
        start_time = time.time()
        loss = criterion(output, output*-1)
        loss.backward()
        backward_time.append(time.time() - start_time)
        
        fmaps = net(input)
        start_time = time.time()
        texture_representation = texture_encoder(fmaps)
        RADAM_time.append(time.time() - start_time)
    
    print('----------', model, device, '----------')    
    print("FORWARD --- %s ms ---" % np.round(np.mean(forward_time)*1000, decimals=4))
    print("BACKWARD--- %s ms ---" % np.round(np.mean(backward_time)*1000, decimals=4))
    print("RADAM   --- %s ms ---" % np.round(np.mean(RADAM_time)*1000, decimals=4))
