import torch

def sample_laplace_noise(loc, scale, shape, dtype, device):
    '''
    https://github.com/pytorch/pytorch/blob/6911ce19d7fcf06e7af241e6494b23acdc320dc4/torch/distributions/laplace.py
    '''
    finfo = torch.finfo(dtype)
    u = torch.zeros(shape, dtype=dtype, device=device).uniform_(finfo.eps - 1, 1)
    return loc - scale * u.sign() * torch.log1p(-u.abs())

def sample_unit_laplace_noise(shape, dtype, device):
    return sample_laplace_noise(0., 1., shape, dtype, device)
