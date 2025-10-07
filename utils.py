import torch
from torch.nn import Parameter
from torch.nn.init import xavier_normal_

def get_param(shape,x):
        param = Parameter(torch.zeros(shape))
        if x == 1 and len(shape) > 1:
                xavier_normal_(param.data)
        
        return param