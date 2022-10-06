import math
import torch
from weight_rewiring import PA_rewiring_torch
import pickle
###############
# ELM
###############
from einops.layers.torch import Rearrange

class ELM_AE():
    def __init__(self, Q, P, N, device):
        self._input_size = P
        self._h_size = Q
        self._device = device
        self.wh = N
        self._activation = torch.sigmoid 
        self.pos_encoding = positionalencoding2d(int(P), int(math.sqrt(N)), int(math.sqrt(N))).to(device)
        
        # self.pos_encoding = torch.reshape(self.pos_encoding, (P, N)).to(device)
        
        
        ########## DEPTH-WISE CONVOLUTION (SPATIAL)
        # self.proj0 = torch.nn.Conv2d(self._input_size, self._input_size, self._h_size,
        #                         groups=self._input_size, padding="same") 
        
        # self.proj0.weight = torch.nn.Parameter(torch.reshape(
        #     my_orth(LCG(self._input_size, self._h_size**2)),
        #                       (self._input_size,1,self._h_size,self._h_size)).to(device))
        
        # self.proj0.bias = torch.nn.Parameter(torch.reshape(
        #     my_orth(LCG(self._input_size, 1)),(self._input_size,)).to(device))
        
        
        ########## POINT-WISE CONVOLUTION (SPECTRAL)
        self.proj = torch.nn.Conv2d(self._input_size, self._h_size, 1, bias=True).to(device)
        self.proj.weight = torch.nn.Parameter(torch.reshape(
            my_orth(LCG(self._input_size, self._h_size)),
                              (self._h_size,self._input_size,1,1)).to(device))
        
        self.proj.bias = torch.nn.Parameter(torch.reshape(
            my_orth(LCG(self._h_size, 1)),(self._h_size,)).to(device))
     
    
    def fit_spectral_AE(self, x):       
        x = torch.add(x, self.pos_encoding) #simpler case, adds pos encoding
        func = Rearrange('c h w -> (h w) c')
        H = func(self._activation(self.proj(x)))
        x= func(x)
        self._beta = torch.linalg.lstsq(H, x).solution #same as # self._beta = torch.mm(x, torch.pinverse(H))
        return self._beta
    
    # def fit_spatial_AE(self, x):       
    #     x = torch.add(x, self.pos_encoding) #simpler case, adds pos encoding
    #     func = Rearrange('c h w -> (h w) c')
    #     H = func(self._activation(self.proj0(x)))
    #     x= func(x)
    #     self._beta = torch.linalg.lstsq(H, x).solution #same as # self._beta = torch.mm(x, torch.pinverse(H))
    #     return self._beta
    
    # def fit_spatiospectral_AE(self, x):
    #     x = torch.add(x, self.pos_encoding) #simpler case, adds pos encoding
    #     func = Rearrange('c h w -> c (h w)')  
    #     Hspc = func(self._activation(self.proj0(x)))
    #     Hspt = func(self._activation(self.proj(x)))
    #     x= func(x)
        
    #     H = Hspt + Hspc + x
        
    #     self._beta = torch.linalg.lstsq(H.t(), x.t()).solution #same as # self._beta = torch.mm(x, torch.pinverse(H))
    #     return self._beta
        
            
def LCG(m, n):
    L = m*n
    
    # if L == 1:
    #     return torch.ones((1,1), dtype=torch.float)
    # else:                    
    #     V = torch.zeros(L, dtype=torch.float)
        
    #     #### Initial approach, according to Jarbas
    #     if L == 4:
    #         V[0] = L+2.0
    #     else:
    #         V[0] = L+1.0        
    #     a = L+2.0
    #     b = L+3.0        
    #     c = L**2.0
        
        #### Better approach
        # V[0] = 1
        # a = 75
        # b = 74   
        # c = (2**16)+1
        
        # for x in range(1, (m*n)):
        #     V[x] = (a*V[x-1]+b) % c

    with open('code_RNN/weights.pkl', 'rb') as f:
        V = pickle.load(f)
        f.close()      
    V = V[0:L]
    
    V = torch.divide(torch.subtract(V, torch.mean(V)), torch.std(V))
 
    return V.reshape((m,n))
    
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


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe