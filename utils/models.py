import torch
import torch.nn.functional as F


def softplus(x):
    return torch.log(torch.exp(x) + 1)

def swish(x):
    '''
    https://arxiv.org/abs/1710.05941
    '''
    return x*torch.sigmoid(x)

def get_nonlinear_func(nonlinearity_type='elu'):
    if nonlinearity_type == 'relu':
        return F.relu
    elif nonlinearity_type == 'elu':
        return F.elu
    elif nonlinearity_type == 'tanh':
        return torch.tanh
    elif nonlinearity_type == 'softplus':
        return F.softplus
    elif nonlinearity_type == 'csoftplus':
        return softplus
    elif nonlinearity_type == 'leaky_relu':
        def leaky_relu(input):
            return F.leaky_relu(input, negative_slope=0.2)
        return leaky_relu
    elif nonlinearity_type == 'swish':
        return swish
    else:
        raise NotImplementedError
