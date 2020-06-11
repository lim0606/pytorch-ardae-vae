'''
copied and modified from https://github.com/CW-Huang/torchkit/blob/master/torchkit/autoencoders.py#L20-L70
'''
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from models.layers import MLP
from models.reparam import NormalDistributionLinear, BernoulliDistribution#, BernoulliDistributionConvTranspose2d
from utils import loss_kld_gaussian, loss_recon_bernoulli_with_logit, normal_energy_func
from utils import logprob_gaussian
from utils import get_nonlinear_func
from utils import conv_out_size, deconv_out_size
import models.layers2 as nn_


def sample_gaussian(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + std * eps

class Encoder(nn.Module):
    def __init__(self,
                 z_dim=32,
                 c_dim=450,
                 act=nn.ELU(),
                 do_center=False,
                 ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.do_center = do_center

        self.enc = nn.Sequential(
            nn_.ResConv2d(1,16,3,2,padding=1,activation=act),
            act,
            nn_.ResConv2d(16,16,3,1,padding=1,activation=act),
            act,
            nn_.ResConv2d(16,32,3,2,padding=1,activation=act),
            act,
            nn_.ResConv2d(32,32,3,1,padding=1,activation=act),
            act,
            nn_.ResConv2d(32,32,3,2,padding=1,activation=act),
            act,
            nn_.Reshape((-1,32*4*4)),
            nn_.ResLinear(32*4*4,c_dim),
            act
        )
        self.reparam = NormalDistributionLinear(c_dim, z_dim)

    def sample(self, mu, logvar):
        return self.reparam.sample_gaussian(mu, logvar)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 28, 28)

        # rescale
        if self.do_center:
            x = 2*x -1

        # enc
        ctx = self.enc(x)
        mu, logvar = self.reparam(ctx)

        # sample
        z = self.sample(mu, logvar)

        return z, mu, logvar

class Decoder(nn.Module):
    def __init__(self,
                 z_dim=32,
                 c_dim=450,
                 act=nn.ELU(),
                 do_m5bias=False,
                 ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.do_m5bias = do_m5bias

        self.dec = nn.Sequential(
            nn_.ResLinear(z_dim,c_dim),
            act,
            nn_.ResLinear(c_dim,32*4*4),
            act,
            nn_.Reshape((-1,32,4,4)),
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
            nn_.ResConv2d(32,32,3,1,padding=1,activation=act),
            act,
            nn_.ResConv2d(32,32,3,1,padding=1,activation=act),
            act,
            nn_.slicer[:,:,:-1,:-1],
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
            nn_.ResConv2d(32,16,3,1,padding=1,activation=act),
            act,
            nn_.ResConv2d(16,16,3,1,padding=1,activation=act),
            act,
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
            nn_.ResConv2d(16,1,3,1,padding=1,activation=act),
        )
        if self.do_m5bias:
            self.dec[-1].conv_01.bias.data.normal_(-3, 0.0001)
        self.reparam = BernoulliDistribution()

    def sample(self, logit):
        return self.reparam.sample_logistic_sigmoid(logit)

    def forward(self, input):
        logit = self.dec(input)

        # sample
        x = self.sample(logit)

        return x, logit

class VAE(nn.Module):
    def __init__(self,
                 energy_func=normal_energy_func,
                 input_height=28,
                 input_channels=1,
                 z_dim=32,
                 c_dim=450,
                 nonlinearity='elu',
                 do_center=False,
                 do_m5bias=False,
                 ):
        super().__init__()
        self.energy_func = energy_func
        self.input_height = input_height
        self.input_channels = input_channels
        self.z_dim = z_dim
        self.latent_dim = self.z_dim # for ais
        self.nonlinearity = nonlinearity
        self.do_center = do_center
        self.do_m5bias = do_m5bias

        assert input_height == 28
        assert input_channels == 1
        assert nonlinearity == 'elu'

        self.encode = Encoder(z_dim=z_dim, c_dim=c_dim, act=nn.ELU(), do_center=do_center)
        self.decode = Decoder(z_dim=z_dim, c_dim=c_dim, act=nn.ELU(), do_m5bias=do_m5bias)

    def loss(self, mu_z, logvar_z, logit_x, target_x, beta=1.0):
        # kld loss
        kld_loss = loss_kld_gaussian(mu_z, logvar_z, do_sum=False)

        # recon loss (likelihood)
        recon_loss = loss_recon_bernoulli_with_logit(logit_x, target_x, do_sum=False)

        # add loss
        loss = recon_loss + beta*kld_loss
        return loss.mean(), recon_loss.mean(), kld_loss.mean()

    def forward(self, input, beta=1.0):
        # init
        batch_size = input.size(0)
        input = input.view(batch_size, self.input_channels, self.input_height, self.input_height)

        # encode
        z, mu_z, logvar_z = self.encode(input)

        # decode
        x, logit_x = self.decode(z)

        # loss
        loss, recon_loss, kld_loss \
                = self.loss(mu_z, logvar_z,
                            logit_x,
                            input,
                            beta=beta,
                            )

        # return
        return x, torch.sigmoid(logit_x), z, loss, recon_loss.detach(), kld_loss.detach()

    def generate(self, batch_size=1):
        # init mu_z and logvar_z (as unit normal dist)
        weight = next(self.parameters())
        mu_z = weight.new_zeros(batch_size, self.z_dim)
        logvar_z = weight.new_zeros(batch_size, self.z_dim)

        # sample z (from unit normal dist)
        z = sample_gaussian(mu_z, logvar_z)  # sample z

        # decode
        output, logit_x = self.decode(z)

        # return
        return output, torch.sigmoid(logit_x), z

    def logprob(self, input, sample_size=128, z=None):
        '''
        input: positive samples
        '''
        # init
        batch_size = input.size(0)
        input = input.view(batch_size, self.input_channels, self.input_height, self.input_height)

        ''' get log q(z|x) '''
        _, mu_qz, logvar_qz = self.encode(input)
        mu_qz = mu_qz.detach().repeat(1, sample_size).view(batch_size, sample_size, self.z_dim)
        logvar_qz = logvar_qz.detach().repeat(1, sample_size).view(batch_size, sample_size, self.z_dim)
        z = self.encode.sample(mu_qz, logvar_qz)
        logposterior = logprob_gaussian(mu_qz, logvar_qz, z, do_unsqueeze=False, do_mean=False)
        logposterior = torch.sum(logposterior.view(batch_size, sample_size, self.z_dim), dim=2) # bsz x ssz

        ''' get log p(z) '''
        # get prior (as unit normal dist)
        mu_pz = input.new_zeros(batch_size, sample_size, self.z_dim)
        logvar_pz = input.new_zeros(batch_size, sample_size, self.z_dim)
        logprior = logprob_gaussian(mu_pz, logvar_pz, z, do_unsqueeze=False, do_mean=False)
        logprior = torch.sum(logprior.view(batch_size, sample_size, self.z_dim), dim=2) # bsz x ssz

        ''' get log p(x|z) '''
        # decode
        logit_x = []
        #for i in range(sample_size):
        for i in range(batch_size):
            _, _logit_x = self.decode(z[i, :, :]) # ssz x zdim
            logit_x += [_logit_x.detach().unsqueeze(0)]
        logit_x = torch.cat(logit_x, dim=0) # bsz x ssz x input_dim
        _input = input.unsqueeze(1).expand(batch_size, sample_size, self.input_channels, self.input_height, self.input_height) # bsz x ssz x input_dim
        loglikelihood = -F.binary_cross_entropy_with_logits(logit_x, _input, reduction='none')
        loglikelihood = torch.sum(loglikelihood.view(batch_size, sample_size, -1), dim=2) # bsz x ssz

        ''' get log p(x|z)p(z)/q(z|x) '''
        logprob = loglikelihood + logprior - logposterior # bsz x ssz
        logprob_max, _ = torch.max(logprob, dim=1, keepdim=True)
        rprob = (logprob - logprob_max).exp() # relative prob
        logprob = torch.log(torch.mean(rprob, dim=1, keepdim=True) + 1e-10) + logprob_max # bsz x 1

        # return
        return logprob.mean()
