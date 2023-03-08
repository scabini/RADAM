from TESTS_finalRADAM import RADAM
import timm
import numpy as np
import torch
import time
import os
import pandas as pd
from sklearn import svm
from joblib import parallel_backend

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"]='0'
os.environ["OMP_NUM_THREADS"]='1'

# model = 'convnext_xlarge_in22k' #select the timm backbone to be coupled with RADAM

backbones = [
            'resnet18', 'resnet50',
            ]

matrix = {}
for model in backbones:
    matrix[model] = []
    for device in ['cpu', 'cuda']:
        forward_time = []
        backward_time = []
        RADAM_time = []
        RADAMSVM_time = []
        finet_time = []
        RADAM_totaltime = []
        for i in range(100):
            seed = i
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            # torch.set_num_threads(1)            
            
            net = timm.create_model(model, features_only=True, pretrained=False, output_stride=8)
            net.eval()
            # device = "cuda"
            # device = "cpu"
            net.to(device)
            act_maps = net(torch.zeros(1,3,224,224).to(device))
            net.eval()
            z = sum([d.size()[1] for d in act_maps])
            half_depth = int(np.round(len(act_maps)/2))
            w = act_maps[half_depth].size()[2]
            h = act_maps[half_depth].size()[3]
            
            
            texture_encoder = RADAM(device, z, (w,h), m=4)
            
            criterion = torch.nn.CrossEntropyLoss()
            
            net_temp = timm.create_model(model, pretrained=False, num_classes=10)
            net_temp.to(device)
            net_temp.train()
            optimizer = torch.optim.AdamW(net_temp.parameters(),                                          
                                          lr=0.001,  weight_decay=0.001) #standard lr=0.001
            
            input = torch.rand(1,3,224,224).to(device) - 0.5
            
            start_time = time.time()
            output = net_temp(input)
            forward_time.append(time.time() - start_time)
            output+=1
        
            start_time = time.time()
            optimizer.zero_grad() 
            loss = criterion(output, output*-1)
            loss.backward()
            optimizer.step() 
            backward_time.append(time.time() - start_time)
            
            finet_time.append(forward_time[-1]+backward_time[-1])
            
            with torch.no_grad():
                SVM = svm.SVC(kernel='linear', max_iter=100000)       
                #train a dummy SVM with 100 samples with z features (RADAM size) and 10 random classes.
                SVM.fit(torch.rand(100, z).cpu().detach().numpy(), torch.round(torch.rand(100)*10).cpu().detach().numpy())
                
                fmaps = net(input)
                texture_representation = texture_encoder(fmaps)#CUDA warmup
                texture_representation=[]
                fmaps = net(input)
                  
                start_time = time.time()
                texture_representation = texture_encoder(fmaps)
                RADAM_time.append(time.time() - start_time)
                preds=SVM.predict(texture_representation.reshape(1,z).cpu().detach().numpy())                 
                RADAMSVM_time.append(time.time() - start_time)
                
                RADAM_totaltime.append(forward_time[-1]+RADAMSVM_time[-1])
                texture_representation+=1
                
            del texture_representation
            del optimizer
            torch.cuda.empty_cache()
            del input
            del texture_encoder
            del act_maps
            del fmaps
            del net
            del net_temp
        
        matrix[model].append([(np.mean(RADAM_time), np.std(RADAM_time)),
                              (np.mean(RADAMSVM_time), np.std(RADAMSVM_time)),
                              (np.mean(forward_time), np.std(forward_time)), 
                              (np.mean(backward_time),  np.std(backward_time)),
                              (np.mean(finet_time),  np.std(finet_time)),
                              (np.mean(RADAM_totaltime),  np.std(RADAM_totaltime))
                              ])
        
        print('----------', model, device, '----------')  
        print("RADAM   ---" , np.round(np.mean(RADAM_time)*1000, decimals=4), np.round(np.std(RADAM_time)*1000, decimals=4))
        print("RADAMSVM---" , np.round(np.mean(RADAMSVM_time)*1000, decimals=4), np.round(np.std(RADAMSVM_time)*1000, decimals=4))
        print("FORWARD ---" , np.round(np.mean(forward_time)*1000, decimals=4), np.round(np.std(forward_time)*1000, decimals=4))
        print("BACKWARD---" , np.round(np.mean(backward_time)*1000, decimals=4), np.round(np.std(backward_time)*1000, decimals=4))
        print("FINE TUN---" , np.round(np.mean(finet_time)*1000, decimals=4), np.round(np.std(finet_time)*1000, decimals=4))
        print("RADAM FE---" , np.round(np.mean(RADAM_totaltime)*1000, decimals=4), np.round(np.std(RADAM_totaltime)*1000, decimals=4))

writer = pd.ExcelWriter('time_experiment.xlsx')
        
df = pd.DataFrame(data=matrix, index=['cpu (RADAM alone), (forw.), (backw.), (fine-t.), (RADAM full)', 
                                      'cuda (RADAM alone), (forw.), (backw.), (fine-t.), (RADAM full)']).T
df.to_excel(writer)
writer.save()






