#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 12:58:54 2022

@author: lucas
"""
import numpy as np
from scipy import stats
from scipy.linalg import orth

def RNN_AE(X,Q=10):
    X = stats.zscore(X) # z-score applied in each feature
    # avg, varr = np.mean(X, axis=1), np.var(X, axis=1)
    
    X = np.nan_to_num(X)
            
    P = X.shape[0] #get the size of input layer
    N = X.shape[1] #get the number of input vectors
   
    # W = LCG(Q,P) #generate the orthonormal weights
    
    W = my_orth(LCG(Q,P)) #generate the orthonormal weights with our own method
    
    # if P > Q:
    #     W = orth(W.transpose(),rcond=0.0).transpose()
    # else:
    #     W = orth(W,rcond=0.0)
    
    # bias = LCG(Q,1) # generate bias weights of the hidden layers
    bias = my_orth(LCG(Q,1)) # generate bias weights of the hidden layers

    bias = np.tile(bias,(1,N)) # Extend the bias matrix to match the demention of Z
     
    Z = np.add(np.matmul(W,X), bias) # W*X + bias
    Z = 1 / (1 + np.exp(-Z)) # activation function

    #calculating the output weights: Beta = XZ'(ZZ' + lamb*eye(Q))^-1
    lamb = 0.001
    Beta = np.dot(np.matmul(X,Z.transpose()), np.linalg.inv(np.matmul(Z,Z.transpose()) + lamb * np.eye(Q)))
    
    erro = np.matmul(Beta, Z) - X
    erro = np.mean(erro**2)
    # print(erro)
    #return the trained output weights
    # Beta = np.nan_to_num(Beta)
    return Beta, erro

### first implementation, had to append at each iteration (costly)
# def LCG(m,n):
#     L = m*n;
#     if L == 1:
#         return np.ones((1,1))        
#     else:
#         if L == 4:#weird case where 4 equal numbers are returned
#             L+=1
            
#         V=np.array([L+1.0], dtype=np.float64);
#         a = L+2.0;
#         b = L+3.0;
#         c = L*L;
#         for x in range(0,(m*n)-1):
#             y = np.float64((a*V[x]+b) % c)
#             V = np.append(V,y) 
  
#         V = stats.zscore(V)
#         V.resize((m,n))
#         return V

### this one is way faster
def LCG(m,n):
    L = m*n;
    if L == 1:
        return np.ones((1,1))        
    else:        
        V = np.zeros(L, dtype=np.float64)
        
        if L == 4:#weird case where 4 equal numbers are returned
            V[0] = L+2.0
        else:
            V[0] = L+1.0
            
        # V=np.array([L+1.0], dtype=np.float64);
        a = L+2.0;
        b = L+3.0;
        c = L*L;
        for x in range(1,(m*n)):
            V[x] = (a*V[x-1]+b) % c
  
        V = stats.zscore(V)
        V.resize((m,n))
        return V
    
    
def my_orth(matrix):
    rows,cols = matrix.shape
    if rows < cols:
        matrix=matrix.transpose()
    
    # Compute the qr factorization
    q, r = np.linalg.qr(matrix)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = np.diag(r, 0)
    ph = np.sign(d)
    q *= ph
 
    if rows < cols:
        q=q.transpose()
        
    return q
    
    
