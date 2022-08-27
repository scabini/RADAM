
import torch
import torch.nn as nn
from scipy import stats

###############
# ELM
###############

class ELM():
    def __init__(self, input_size, h_size, device=None):
        self._input_size = input_size
        self._h_size = h_size
        self._output_size = input_size
        self._device = device
        self._lambda = 0.001
        
        if(device):
            self._alpha = my_orth(LCG(self._input_size, self._h_size)).to(device)
            self._bias = my_orth(LCG(self._h_size, 1)).to(device)
            self._eye = torch.eye(self._h_size, dtype=torch.float).to(device)
        else:
            self._alpha = my_orth(LCG(self._input_size, self._h_size))
            self._bias = my_orth(LCG(self._h_size, 1))
            self._eye = torch.eye(self._h_size, dtype=torch.float)
            
            
        self._activation = torch.sigmoid
        
    # def predict(self, x):
    #     h = self._activation(torch.add(x.mm(self._alpha), self._bias))
    #     out = h.mm(self._beta)
        
        # return out
    
    def fit(self, x):
        
        bias = torch.tile(self._bias, (1,x.shape[1]))
        temp = torch.mm(self._alpha.t(), x)
        H = self._activation(torch.add(temp, bias))
        
        # self._beta = torch.mm(torch.mm(x, H.t()), torch.linalg.inv(torch.mm(H, H.t()) + self._lambda * self._eye))
        
        self._beta = torch.mm(x, torch.pinverse(H))
        
    # def evaluate(self, x, t):
    #     y_pred = self.predict(x)
    #     acc = torch.sum(torch.argmax(y_pred, dim=1) == torch.argmax(t, dim=1)).item() / len(t)
    #     return acc
    
    def get_output_weights(self):
        return self._beta
        
        

#####################
# Helper Functions
#####################
# def to_onehot(batch_size, num_classes, y, device):
#     # One hot encoding buffer that you create out of the loop and just keep reusing
#     y_onehot = torch.FloatTensor(batch_size, num_classes).to(device)
#     #y = y.type(dtype=torch.long)
#     y = torch.unsqueeze(y, dim=1)
#     # In your for loop
#     y_onehot.zero_()
#     y_onehot.scatter_(1, y, 1)

#     return y_onehot

def LCG(m, n):
    L = m*n
    if L == 1:
        return torch.ones((1,1), dtype=torch.float)
    else: 
                    
        V = torch.zeros(L, dtype=torch.float)
        if L == 4:
            V[0] = L+2.0
        else:
            V[0] = L+1.0
        
        a = L+2.0
        b = L+3.0
        
        c = L**2.0
        
        for x in range(1, (m*n)):
            V[x] = (a*V[x-1]+b) % c
            
        
        V = torch.divide(torch.subtract(V, torch.mean(V)), torch.var(V))
        # print(V)
        V = V.reshape((m,n))
        
        return V
    
def my_orth(tensor):
    rows = tensor.size(0)
    cols = tensor.numel() // rows
    flattened = torch.reshape(tensor, (rows, cols))

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    q, r = torch.linalg.qr(flattened)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph

    if rows < cols:
        q.t_()
        
    return q
    return tensor

