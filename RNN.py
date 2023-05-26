import math
import torch
import pickle


class RAE():
    def __init__(self, Q, P, N, device, pos_encoding, seed):
        self._input_size = P
        self._h_size = Q
        self._device = device
        self.pos_encoding=pos_encoding
        # self._lambda = 0.001
        
        # self._alpha = torch.ones(self._h_size, self._input_size).to(device)
        self._alpha = make_orthogonal(LCG(self._h_size, self._input_size, seed)).to(device)
        # self._bias = LCG(self._h_size, 1)
        # self._bias = torch.ones((self._h_size, 1))
        
        if self.pos_encoding:
            window_size = math.sqrt(N)
            self.encoding = positionalencoding2d(int(P), int(window_size), int(window_size))
            self.encoding = torch.reshape(self.encoding, (P, N)).to(device)
    
        # self.bias = torch.tile(self._bias, (1,N)).to(device)
        # self._eye = torch.eye(self._h_size, dtype=torch.float).to(device)
 
        self._activation = torch.sigmoid
        
    # def predict(self, x):
    #     h = self._activation(torch.add(x.mm(self._alpha), self._bias))
    #     out = h.mm(self._beta)
        
        # return out
    # def project(self, x):       
    #     x = torch.add(x, self.pos_encoding) #simpler case, adds pos encoding
    #     temp = torch.mm(self._alpha, x)
    #     # temp = torch.mm(self._alpha, torch.cat((x, self.pos_encoding))) # new case, concatenate the encoding

    #     return self._activation(torch.add(temp, self.bias))
    
    
    def fit_AE(self, x):     
        if self.pos_encoding:
            x = torch.add(x, self.encoding) #simpler case, adds pos encoding
        # temp = torch.add(temp, self.pos_encoding)
        # temp = torch.mm(self._alpha, torch.cat((x, self.pos_encoding))) # new case, concatenate the encoding

        # H = self._activation(torch.add(temp, self.bias))
        H = self._activation(torch.mm(self._alpha, x))
        # self._beta = torch.mm(torch.mm(x, H.t()), torch.linalg.inv(torch.mm(H, H.t()) + self._lambda * self._eye))
        
        # self._beta = torch.mm(x, torch.pinverse(H))
        self._beta = torch.linalg.lstsq(H.t(), x.t()).solution
        return self._beta
    
    # def fit_REG(self, x, target_):       
    #     x = torch.add(x, self.pos_encoding) #simpler case, adds pos encoding
    #     target = torch.reshape(x[target_,:], (1, x.size()[1]))
    #     x = torch.cat((x[:target_],x[target_+1:]))        
    #     temp = torch.mm(self._alpha[:,:-1], x)
    #     H = self._activation(torch.add(temp, self.bias))        
        
    #     self._beta = torch.mm(target, torch.pinverse(H))
    #     return self._beta
    
    # def fit_agg(self, x, target_):       
    #     # x = torch.add(x, self.pos_encoding) #simpler case, adds pos encoding
    #     temp = torch.mm(self._alpha, x)
    #     # temp = torch.mm(self._alpha, torch.cat((x, self.pos_encoding))) # new case, concatenate the encoding

    #     # H = self._activation(torch.add(temp, self.bias))
    #     H = self._activation(temp)
        
    #     # self._beta = torch.mm(torch.mm(x, H.t()), torch.linalg.inv(torch.mm(H, H.t()) + self._lambda * self._eye))
        
    #     # self._beta = torch.mm(target_, torch.pinverse(H))
    #     self._beta = torch.linalg.lstsq(H.t(), target_.t()).solution
    #     return self._beta
    
    # def evaluate(self, x, t):
    #     y_pred = self.predict(x)
    #     acc = torch.sum(torch.argmax(y_pred, dim=1) == torch.argmax(t, dim=1)).item() / len(t)
    #     return acc
    
    # def get_output_weights(self):
    #     return self._beta
        
def LCG(m, n, seed):
    L = m*n
    
    if L == 1:
        return torch.ones((1,1), dtype=torch.float)
    else:      

        with open('RAE_LCG_weights.pkl', 'rb') as f:
            V = pickle.load(f)
            f.close()      
        V = V[seed:L+seed]
              
        ##### If you want to generate the weights everytime, instead of loading
        #       from our file, just uncomment these lines below and remove the above ones
        
        #V = torch.zeros(L, dtype=torch.float)    
        #V[0] = 0
        #a = 75
        #b = 74   
        #c = (2**16)+1        
        #for x in range(1, (m*n)):
        #   V[x] = (a*V[x-1]+b) % c

          
        #Always keep the zscore normalization for our LCG weights
        V = torch.divide(torch.subtract(V, torch.mean(V)), torch.std(V))
 
    return V.reshape((m,n))
    
def make_orthogonal(tensor):
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
    # return tensor


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    # if d_model % 4 != 0:
    #     raise ValueError("Cannot use sin/cos positional encoding with "
    #                      "odd dimension (got dim={:d})".format(d_model))
    d_model_orig = d_model
    if d_model % 4 != 0:
        d_model = d_model+2   # Round up to the nearest multiple of 4
    else:
        d_model = d_model_orig    
        
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

    return pe[:d_model_orig, :, :]
