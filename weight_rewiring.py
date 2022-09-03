# -*- coding: utf-8 -*-
"""
PA rewiring method

@author: scabini
"""
import torch
import numpy as np
  
def rewiring_np(weights, seed):
    dimensions = weights.shape     
    if seed:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng() 
    st = np.zeros(dimensions[1])
    for neuron in range(1,dimensions[0]): #loops over output neurons from 2 to n-1 
        st = st + weights[neuron-1]   #compute the temporary strength     
        P = st + np.abs(np.min(st)) + 1 #make the distribution positive and avoid null probability       
        P = P / np.sum(P) #pdf from negative to positive     
        targets = rng.choice(a=[i for i in range(dimensions[1])], replace=False, 
                                      size=dimensions[1],p=P) 
        edges_to_rewire = np.argsort(weights[neuron]) #sort the edges to be rewired
        weights[neuron, targets] = weights[neuron, edges_to_rewire] #rewiring             
    return weights

def rewiring_torch(weights, seed):
    if seed:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()        
    st = torch.zeros(weights.size(1))
    with torch.no_grad():
        dimensions = weights.shape 
        for neuron in range(1,dimensions[0]):  
            st = st + weights[neuron-1]
            P = st + torch.abs(torch.min(st)) + 1 #the +1 is to ensure no zero values
            P = P / torch.sum(P)
            targets = rng.choice(a=[i for i in range(dimensions[1])], replace=False,
                                          size=dimensions[1],p=P.cpu().detach().numpy()) 
            edges_to_rewire = torch.argsort(weights[neuron])
            weights[neuron, targets] = weights[neuron, edges_to_rewire]                
    return weights

def rewiring_torch_deterministic(weights):           
    st = torch.zeros(weights.size(1))
    with torch.no_grad():
        dimensions = weights.shape 
        for neuron in range(1,dimensions[0]):  
            st = st + weights[neuron-1]
            targets = torch.argsort(st)
            edges_to_rewire = torch.argsort(-weights[neuron])
            
            weights[neuron, targets] = weights[neuron, edges_to_rewire]                
    return weights

### Numpy version: has better numerical precision than the torch version; 
#   however, it creates a copy of the weight tensor
def PA_rewiring_np(weights, seed=False): 
    if weights.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported") 
    output_neurons = weights.size(0)
    input_neurons = weights.numel() // output_neurons
    dimensions = weights.shape    
    weights_out = weights.cpu().detach().numpy()        
    weights_out = weights_out.reshape((output_neurons, input_neurons))    
    rewiring_np(weights_out, seed) #rewire input neurons
    rewiring_np(np.transpose(weights_out), seed) #rewire output neurons    
    weights_out = weights_out.reshape((dimensions))         
    weights_out=torch.from_numpy(weights_out)
    with torch.no_grad():
        weights.view_as(weights_out).copy_(weights_out)      
    return weights  #cast the obtained tensor into the input tensor 'weights'

### Pytorch implementation: looses precision on large sums compared to
#   numpy, i.e., the strength calculations will be different.
#   I recommend using the np version (but check efficiency)
def PA_rewiring_torch(weights, seed=False, stochastic=True): 
    if weights.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")
    with torch.no_grad():  
        output_neurons = weights.size(0)
        input_neurons = weights.numel() // output_neurons    
        dimensions = weights.shape             
        weights = weights.reshape((output_neurons, input_neurons))
        if stochastic:
            rewiring_torch(weights, seed)   
            rewiring_torch(torch.transpose(weights, 0, 1), seed) 
        else:
            rewiring_torch_deterministic(weights)   
            rewiring_torch_deterministic(torch.transpose(weights, 0, 1))
            
        weights = weights.reshape((dimensions))          
    return weights  #cast the obtained tensor into the input tensor 'weights'

def stabilize_strength(initializer, weights, K=100): #how much stable? =) increase K!
### random search to minimize strength variance
#initializer is a lambda function that receives the weights, eg:
# initializer = lambda w : torch.nn.init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu')
#or simply:
# initializer = torch.nn.init.orthogonal_
#weights is a torch weight matrix taken from a layer
    maximus = np.Infinity 
    dimensions = weights.shape   
    output_neurons = weights.size(0)
    input_neurons = weights.numel() // output_neurons  
    weights_out = torch.empty(dimensions)
    for i in range(K): 
        initializer(weights)                              
        weights = weights.reshape((output_neurons, input_neurons))
        localmax = torch.mean(torch.hstack((torch.var(torch.sum(weights, dim=0) ), torch.var(torch.sum(weights, dim=1) ))))     
        if localmax < maximus:
            maximus = localmax
            with torch.no_grad(): 
                weights_out.view_as(weights).copy_(weights)   
                
    with torch.no_grad(): 
        weights.view_as(weights_out.reshape(dimensions)).copy_(weights_out.reshape(dimensions))
       
    return weights

### HOW TO USE:
# eg. rewiring all layers of a conv2d model:

# for m in model.modules():
#     if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
#         PA_rewiring_np(m.weight)
    

    
