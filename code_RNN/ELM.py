
import torch
from weight_rewiring import PA_rewiring_torch
###############
# ELM
###############

class ELM_AE():
    def __init__(self, Q, P, N, device):
        self._input_size = P
        self._h_size = Q
        self._output_size = P
        self._device = device
        self._lambda = 0.001
        
        self._alpha = my_orth(LCG(self._h_size, self._input_size)).to(device)
        # self._alpha = PA_rewiring_torch(my_orth(LCG(self._h_size, self._input_size)),stochastic=False).to(device)
        # self._bias = my_orth(LCG(self._h_size, 1))
        self._bias = torch.ones((self._h_size, 1))
        self.bias = torch.tile(self._bias, (1,N)).to(device)
        self._eye = torch.eye(self._h_size, dtype=torch.float).to(device)
 
        self._activation = torch.sigmoid
        
    # def predict(self, x):
    #     h = self._activation(torch.add(x.mm(self._alpha), self._bias))
    #     out = h.mm(self._beta)
        
        # return out
    
    def fit(self, x):       
        
        temp = torch.mm(self._alpha, x)
        H = self._activation(torch.add(temp, self.bias))
        
        # self._beta = torch.mm(torch.mm(x, H.t()), torch.linalg.inv(torch.mm(H, H.t()) + self._lambda * self._eye))
        
        self._beta = torch.mm(x, torch.pinverse(H))
        return self._beta
    
    # def evaluate(self, x, t):
    #     y_pred = self.predict(x)
    #     acc = torch.sum(torch.argmax(y_pred, dim=1) == torch.argmax(t, dim=1)).item() / len(t)
    #     return acc
    
    def get_output_weights(self):
        return self._beta
        
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
            
        
        V = torch.divide(torch.subtract(V, torch.mean(V)), torch.std(V))
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

