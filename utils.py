import torch
from torch.nn import Parameter
from torch.nn.init import xavier_normal_

def get_param(shape):
        param = Parameter(torch.Tensor(*shape))
        xavier_normal_(param.data)
        return param