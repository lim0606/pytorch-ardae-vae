'''
miscellaneous functions: prob
'''
import os
import datetime
import math

import numpy as np

import torch
import torch.nn.functional as F
#from torch.autograd import Variable
#from torch.distributions import Categorical, Normal


''' for vae '''
#def binary_cross_entropy_with_logits(logit, target):
#    zeros = logit.new_zeros(logit.size())
#    return torch.max(logit, zeros) - logit * target + (1 + (-logit.abs()).exp()).log()

def loss_recon_bernoulli_with_logit(logit, x, do_sum=True):
    # p = recon prob
    if do_sum:
        return  F.binary_cross_entropy_with_logits(logit, x, reduction='sum')
        #return  torch.sum(binary_cross_entropy_with_logits(logit, x))
    else:
        batch_size = x.size(0)
        cross_entropy = F.binary_cross_entropy_with_logits(logit, x, reduction='none')
        #cross_entropy = binary_cross_entropy_with_logits(logit, x)
        return torch.sum(cross_entropy.view(batch_size, -1), dim=1)

def loss_recon_bernoulli(p, x):
    # p = recon prob
    return F.binary_cross_entropy(p, x, size_average=False)

def loss_recon_gaussian(mu, logvar, x, const=None, do_sum=True):
    # https://math.stackexchange.com/questions/1307381/logarithm-of-gaussian-function-is-whether-convex-or-nonconvex
    # mu, logvar = nomral distribution
    recon_loss_element = logvar + (x - mu)**2 / logvar.exp() + math.log(2.*math.pi)

    # add const (can be used in change of variable)
    if const is not None:
        recon_loss_element += const

    # do sum
    if do_sum:
        recon_loss = torch.sum(recon_loss_element) * 0.5
        return recon_loss
    else:
        batch_size = recon_loss_element.size(0)
        recon_loss_element = torch.sum(recon_loss_element.view(batch_size, -1), 1) * 0.5
        return recon_loss_element

def loss_recon_gaussian_w_fixed_var(mu, x, std=1.0, const=None, do_sum=True, add_logvar=True):
    # init var, logvar
    var = std**2
    logvar = math.log(var)

    # estimate loss per element
    if add_logvar:
        recon_loss_element = logvar + (x - mu)**2 / var + math.log(2.*math.pi)
    else:
        recon_loss_element = (x - mu)**2 / var + math.log(2.*math.pi)

    # add const (can be used in change of variable)
    if const is not None:
        recon_loss_element += const

    # do sum
    if do_sum:
        recon_loss = torch.sum(recon_loss_element) * 0.5
        return recon_loss
    else:
        batch_size = recon_loss_element.size(0)
        recon_loss_element = torch.sum(recon_loss_element.view(batch_size, -1), 1) * 0.5
        return recon_loss_element

def loss_kld_gaussian(mu, logvar, do_sum=True):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = 1 + logvar - mu.pow(2) - logvar.exp()

    # do sum
    if do_sum:
        KLD = torch.sum(KLD_element) * -0.5
        return KLD
    else:
        batch_size = KLD_element.size(0)
        KLD_element = torch.sum(KLD_element.view(batch_size, -1), 1) * -0.5
        return KLD_element

def loss_kld_gaussian_vs_gaussian(mu1, logvar1, mu2, logvar2, do_sum=True):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    # log(sigma2) - log(sigma1) + 0.5 * (sigma1^2 + (mu1 - mu2)^2) / sigma2^2 - 0.5
    # 0 - log(sigma1) + 0.5 * (sigma1^2 + mu1^2)  - 0.5
    # 0 - log(sigma1) + 0.5 * sigma1^2 + 0.5 * mu1^2  - 0.5
    # 0 - 0.5 * log(sigma1^2) + 0.5 * sigma1^2 + 0.5 * mu1^2  - 0.5
    # log(sigma2) - log(sigma1) + 0.5 * (sigma1^2 + (mu1 - mu2)^2) / sigma2^2 - 0.5
    KLD_element = - logvar2 + logvar1 - (logvar1.exp() + (mu1 - mu2)**2) / logvar2.exp() + 1.

    # do sum
    if do_sum:
        KLD = torch.sum(KLD_element) * -0.5
        return KLD
    else:
        batch_size = KLD_element.size(0)
        KLD_element = torch.sum(KLD_element.view(batch_size, -1), 1) * -0.5
        return KLD_element
