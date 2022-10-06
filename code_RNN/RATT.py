import math
import torch
from weight_rewiring import PA_rewiring_torch
import pickle
###############
# ELM
###############
from einops.layers.torch import Rearrange

class randomized_attention():
    def __init__(self, Q, P, N, device):
        self.z = P
        self.d = Q**0.5
        self._device = device
        self.n = N
        self._activation = torch.sigmoid 
        self.pos_encoding = positionalencoding2d(int(P), int(math.sqrt(N)), int(math.sqrt(N)))
        self.pos_encoding = torch.reshape(self.pos_encoding, (P, N)).to(device)
        
        self.Wq = my_orth(LCG(self.z, int(self.d**2), seed=0)).to(device)
        self.Wk = my_orth(LCG(self.z, int(self.d**2), seed=self.z *int(self.d**2))).to(device)
        self.Wv = my_orth(LCG(self.z, int(self.d**2), seed=2*self.z *int(self.d**2))).to(device)
         
    def self_att(self, x):       
        x = torch.add(x, self.pos_encoding).t() #simpler case, adds pos encoding
        # func = Rearrange('c h w -> (h w) c')
        # x= func(x)
        
        Q = torch.mm(x, self.Wq) 
        K = torch.mm(x, self.Wk) 
        V = torch.mm(x, self.Wv) 
        A = self._activation(torch.mm(torch.softmax(torch.div(torch.mm(Q, K.t()), self.d), dim=0), V))
        
        self._beta = torch.linalg.lstsq(A, x).solution #same as # self._beta = torch.mm(x, torch.pinverse(H))
        return self._beta
  
            
def LCG(m, n, seed=2):
    L = m*n
    
    # if L == 1:
    #     return torch.ones((1,1), dtype=torch.float)
    # else:                    
    #     V = torch.zeros(L, dtype=torch.float)
        
    #     #### Initial approach, according to Jarbas
    #     V[0] = seed
        
    #     a = L+2.0
    #     b = L+3.0        
    #     c = L**2.0
        
    #     ### Better approach
    #     # V[0] = 1
    #     # a = 75
    #     # b = 74   
    #     # c = (2**16)+1
        
    #     for x in range(1, (m*n)):
    #         V[x] = (a*V[x-1]+b) % c

    with open('code_RNN/weights.pkl', 'rb') as f:
        V = pickle.load(f)
        f.close()      
    V = V[seed:L+seed]
    
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