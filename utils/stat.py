import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def shuffle(z):
    batch_size = z.size(0)
    z_dim = z.size(1)
    indices = [torch.from_numpy(np.random.permutation(batch_size)).to(z.device) for i in range(z_dim)]
    new_z = [z[:,i:i+1].index_select(0, indices[i]) for i in range(z_dim)]
    new_z = torch.cat(new_z, dim=1)
    return new_z

def loss_entropy_gaussian(mu, logvar, do_sum=True):
    # mu, logvar = nomral distribution
    entropy_loss_element = logvar + 1. + math.log(2.*math.pi)

    # do sum
    if do_sum:
        entropy_loss = torch.sum(entropy_loss_element) * 0.5
        return entropy_loss
    else:
        #entropy_loss_element = torch.sum(entropy_loss_element, 1) * 0.5 + (1. + math.log(2.*math.pi)) * 0.5
        entropy_loss_element = entropy_loss_element * 0.5
        return entropy_loss_element

def prob_gaussian(mu, logvar, z, eps=1e-6, do_unsqueeze=True, do_mean=True):
    '''
    Inputs: 
        z: b1 x nz
        mu, logvar: b2 x nz
    Outputs:
        prob: b1 x nz
    '''
    if do_unsqueeze:
        z = z.unsqueeze(1)
        mu = mu.unsqueeze(0)
        logvar = logvar.unsqueeze(0)

    var = logvar.exp() + eps
    std = torch.sqrt(var) + eps

    prob = torch.exp(- 0.5 * (z - mu)**2 / var) / std / math.sqrt(2.*math.pi)

    if do_mean:
        assert do_unsqueeze
        prob = torch.mean(prob, dim=1)

    return prob

def loss_marginal_entropy_gaussian(mu, logvar, z, do_sum=True):
    marginal_entropy_loss_element = -torch.log(prob_gaussian(mu, logvar, z))

    # do sum
    if do_sum:
        marginal_entropy_loss = torch.sum(marginal_entropy_loss_element)
        return marginal_entropy_loss
    else:
        #marginal_entropy_loss_element = torch.sum(marginal_entropy_loss_element, 1)
        return marginal_entropy_loss_element

def logprob_gaussian(mu, logvar, z, do_unsqueeze=True, do_mean=True):
    '''
    Inputs: 
        z: b1 x nz
        mu, logvar: b2 x nz
    Outputs:
        prob: b1 x nz
    '''
    if do_unsqueeze:
        z = z.unsqueeze(1)
        mu = mu.unsqueeze(0)
        logvar = logvar.unsqueeze(0)

    neglogprob = (z - mu)**2 / logvar.exp() + logvar + math.log(2.*math.pi)
    logprob = - neglogprob*0.5

    if do_mean:
        assert do_unsqueeze
        logprob = torch.mean(logprob, dim=1)

    return logprob

def loss_approx_marginal_entropy_gaussian(mu, logvar, z, do_sum=True):
    marginal_entropy_loss_element = -logprob_gaussian(mu, logvar, z)

    # do sum
    if do_sum:
        marginal_entropy_loss = torch.sum(marginal_entropy_loss_element)
        return marginal_entropy_loss
    else:
        #marginal_entropy_loss_element = torch.sum(marginal_entropy_loss_element, 1)
        return marginal_entropy_loss_element

def logprob_gaussian_w_fixed_var(mu, z, std=1.0, do_unsqueeze=True, do_mean=True):
    '''
    Inputs: 
        z: b1 x nz
        mu, logvar: b2 x nz
    Outputs:
        prob: b1 x nz
    '''
    # init var, logvar
    var = std**2
    logvar = math.log(var)

    if do_unsqueeze:
        z = z.unsqueeze(1)
        mu = mu.unsqueeze(0)
        #logvar = logvar.unsqueeze(0)

    neglogprob = (z - mu)**2 / var + logvar + math.log(2.*math.pi)
    logprob = - neglogprob*0.5

    if do_mean:
        assert do_unsqueeze
        logprob = torch.mean(logprob, dim=1)

    return logprob

'''
copied and modified from https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
'''
def get_covmat(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.
            rowvar == True:  m: dim x batch_size
            rowvar == False: m: batch_size x dim
    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()
