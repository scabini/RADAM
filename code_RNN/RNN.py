#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 12:58:54 2022

@author: lucas
"""
import numpy as np
from scipy import stats
from scipy.linalg import orth
import sklearn


class RNN_AE(): #NEW TORCH VERSION
    def __init__(self, Q, P, N):
        super(RNN_AE, self).__init__() 
        self.Q=Q
        self.bias = np.ones((Q,1)) # generate bias weights of the hidden layers
        # self.bias = my_orth(LCG(Q,1)) # generate bias weights of the hidden layers
        self.bias = np.tile(self.bias,(1,N)) # Extend the bias matrix to match the demention of Z
        
        # W = LCG(Q,P+1) #generate the orthonormal weights   
        self.W = my_orth(LCG(Q,P)) #generate the orthonormal weights with our own method
        # W = my_rand(Q,P) #generate the orthonormal weights with our own method
        # W = my_orth(my_rand(Q,P+1)) #generate the orthonormal weights with our own method
        self.eye = np.eye(self.Q)
        self.lamb = 0.001
        
    def fit(self, X, axis=None):        
        Z = np.add(np.matmul(self.W,X), self.bias) # W*X + bias
        Z = 1 / (1 + np.exp(-Z)) # activation function
     
        #calculating the output weights: Beta = XZ'(ZZ' + lamb*eye(Q))^-1
        
        Beta = np.matmul(np.matmul(X,Z.transpose()), np.linalg.inv(np.matmul(Z,Z.transpose()) + self.lamb * self.eye))
        
        # erro = np.matmul(Beta, Z) - X
        # erro = np.mean(erro**2)
        # erro=0
        # print(Beta.shape)
        return Beta

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
        a = L+2.0; #a
        b = L+3.0; #b
        c = L*L; #m
        
        # V[0] = 1
        # a = 48271
        # b = 1   
        # c = (2**31)-1
        
        for x in range(1,(m*n)):
            V[x] = (a*V[x-1]+b) % c
  
        V = stats.zscore(V)
        # print(np.min(V), np.max(V))
        V.resize((m,n))
        return V

# GLOBAL_WEIGHTS_ = LCG(100*2, 512+100)

# def my_rand(m,n):
#     np.random.seed(66699)
#     return np.random.normal(loc=0.0, scale=1.0, size=(m,n))
    # return np.random.normal(loc=0.0, scale=np.sqrt(2/(m+n)), size=(m,n))
    
    
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

# def pi_initializer(m,n):
#     if m*n == 1:
#         return np.ones((1,1), dtype=np.float64)
#     else:
#         digits = [i/9.0 for i in list(pi_digits(m*n))]
#         digits = stats.zscore(np.asarray(digits, dtype=np.float64)).reshape((m,n))
#         # print(digits)
#         return digits
    
# def pi_digits(x):
#     """Generate x digits of Pi."""
#     k,a,b,a1,b1 = 2,4,1,12,4
#     while x > 0:
#         p,q,k = k * k, 2 * k + 1, k + 1
#         a,b,a1,b1 = a1, b1, p*a + q*a1, p*b + q*b1
#         d,d1 = a/b, a1/b1
#         while d == d1 and x > 0:
#             yield int(d)
#             x -= 1
#             a,a1 = 10*(a % b), 10*(a1 % b1)
#             d,d1 = a/b, a1/b1
            
            
            
