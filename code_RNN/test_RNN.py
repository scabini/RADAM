#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 23:06:35 2022

@author: lucas
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from RNN import RNN_AE, LCG, LCG_evolved
import numpy as np
import time
# np.random.seed(666999)

w= 1
h = 512
z = 512

# exemplo de matriz de features de 10 linhas (features) x 150 colunas (amostras/vetores)
# X = np.random.rand(w*h,z) #spatial
# X = np.random.rand(z, w*h) #spectral

# #passa para o RNN auto encoder que vai retornar a matriz de pesos de saida treinados Beta
# Beta = RNN_AE(X, Q=1) 
# feature_vector = Beta.reshape((Beta.shape[0]*Beta.shape[1]))


######### improving LCG
# start = time.time()
# a = LCG(z, w*h) #this is the old implementation
# print('LCG took', time.time()-start, ' seconds')

# start = time.time()
# b = LCG_evolved(z, w*h)
# print('LCG_evolved took', time.time()-start, ' seconds')

# c = a -b