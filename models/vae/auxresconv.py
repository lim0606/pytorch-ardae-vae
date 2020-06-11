import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from models.layers import Identity, MLP
from models.reparam import NormalDistributionLinear, BernoulliDistributionLinear, BernoulliDistributionConvTranspose2d
from utils import loss_kld_gaussian, loss_kld_gaussian_vs_gaussian, loss_recon_gaussian, loss_recon_bernoulli_with_logit, normal_energy_func
from utils import logprob_gaussian
from utils import get_nonlinear_func
from utils import conv_out_size, deconv_out_size
import models.layers2 as nn_
from models.vae.resconv import Decoder


def sample_gaussian(mu, logvar, _std=1.):
    if _std is None:
        _std = 1.
    std = _std*torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + std * eps

class InputEncoder(nn.Module):
    def __init__(self,
                 c_dim=450,
                 act=nn.ELU(),
                 do_center=False,
                 ):
        super().__init__()
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

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 28, 28)

        # rescale
        if self.do_center:
            x = 2*x -1

        # enc
        ctx = self.enc(x)

        return ctx

class AuxEncoder(nn.Module):
    def __init__(self,
                 z0_dim=32,
                 c_dim=450,
                 act=nn.ELU(),
                 do_center=False,
                 clip_logvar=None,
                 ):
        super().__init__()
        self.z0_dim = z0_dim
        self.c_dim = c_dim
        self.do_center = do_center
        self.clip_logvar = clip_logvar

        #self.enc = nn.Sequential(
        #    nn_.ResConv2d(1,16,3,2,padding=1,activation=act),
        #    act,
        #    nn_.ResConv2d(16,16,3,1,padding=1,activation=act),
        #    act,
        #    nn_.ResConv2d(16,32,3,2,padding=1,activation=act),
        #    act,
        #    nn_.ResConv2d(32,32,3,1,padding=1,activation=act),
        #    act,
        #    nn_.ResConv2d(32,32,3,2,padding=1,activation=act),
        #    act,
        #    nn_.Reshape((-1,32*4*4)),
        #    nn_.ResLinear(32*4*4,c_dim),
        #    act
        #)
        self.reparam = NormalDistributionLinear(c_dim, z0_dim, nonlinearity=clip_logvar)

    def sample(self, mu, logvar, _std=1.):
        return sample_gaussian(mu, logvar, _std=_std)

    #def forward(self, x,, _std=1.):
        #batch_size = x.size(0)
        #x = x.view(batch_size, 1, 28, 28)

        ## rescale
        #if self.do_center:
        #    x = 2*x -1
    def forward(self, ctx, _std=1.):
        batch_size = ctx.size(0)

        # enc
        #ctx = self.enc(x)
        mu, logvar = self.reparam(ctx)

        # sample
        z0 = self.sample(mu, logvar, _std=_std)

        return z0, mu, logvar#, ctx

class Encoder(nn.Module):
    def __init__(self,
                 z0_dim=32,
                 z_dim=32,
                 c_dim=450,
                 act=nn.ELU(),
                 do_center=False,
                 clip_logvar=None,
                 ):
        super().__init__()
        self.z0_dim = z0_dim
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.do_center = do_center
        self.clip_logvar = clip_logvar

        #self.enc = nn.Sequential(
        #    nn_.ResConv2d(1,16,3,2,padding=1,activation=act),
        #    act,
        #    nn_.ResConv2d(16,16,3,1,padding=1,activation=act),
        #    act,
        #    nn_.ResConv2d(16,32,3,2,padding=1,activation=act),
        #    act,
        #    nn_.ResConv2d(32,32,3,1,padding=1,activation=act),
        #    act,
        #    nn_.ResConv2d(32,32,3,2,padding=1,activation=act),
        #    act,
        #    nn_.Reshape((-1,32*4*4)),
        #    nn_.ResLinear(32*4*4,c_dim),
        #    act
        #)
        self.fc = nn.Sequential(
                nn.Linear(c_dim + z0_dim, c_dim, bias=True),
                act,
                )
        self.reparam = NormalDistributionLinear(c_dim, z_dim, nonlinearity=clip_logvar)

    def sample(self, mu_z, logvar_z):
        return self.reparam.sample_gaussian(mu_z, logvar_z)

    #def forward(self, x, z0, nz=1):
    #    #batch_size = x.size(0)
    #    #x = x.view(batch_size, 1, 28, 28)
    #    ## rescale
    #    #if self.do_center:
    #    #    x = 2*x -1
    def forward(self, ctx, z0, nz=1):
        batch_size = ctx.size(0)
        assert z0.size(0) == batch_size*nz

        # enc
        #ctx = self.enc(x)

        # view
        ctx = ctx.unsqueeze(1).expand(-1, nz, -1).contiguous()
        ctx = ctx.view(batch_size*nz, -1)

        # concat
        ctxz0 = torch.cat([ctx, z0], dim=1)

        # forward
        h = self.fc(ctxz0)
        mu, logvar = self.reparam(h)

        # sample
        z = self.sample(mu, logvar)

        return z, mu, logvar, h

class AuxDecoder(nn.Module):
    def __init__(self,
                 z_dim=32,
                 c_dim=450,
                 z0_dim=100,
                 act=nn.ELU(),
                 do_center=False,
                 ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.z0_dim = z0_dim
        self.do_center = do_center

        #self.enc = nn.Sequential(
        #    nn_.ResConv2d(1,16,3,2,padding=1,activation=act),
        #    act,
        #    nn_.ResConv2d(16,16,3,1,padding=1,activation=act),
        #    act,
        #    nn_.ResConv2d(16,32,3,2,padding=1,activation=act),
        #    act,
        #    nn_.ResConv2d(32,32,3,1,padding=1,activation=act),
        #    act,
        #    nn_.ResConv2d(32,32,3,2,padding=1,activation=act),
        #    act,
        #    nn_.Reshape((-1,32*4*4)),
        #    nn_.ResLinear(32*4*4,c_dim),
        #    act
        #)
        self.fc = nn.Sequential(
                nn.Linear(c_dim + z_dim, c_dim, bias=True),
                act,
                )
        self.reparam = NormalDistributionLinear(c_dim, z0_dim)

    def sample(self, mu, logvar):
        return self.reparam.sample_gaussian(mu, logvar)

    #def forward(self, x, z, nz=1):
        #batch_size = x.size(0)
        #x = x.view(batch_size, 1, 28, 28)
        ## rescale
        #if self.do_center:
        #    x = 2*x -1
    def forward(self, ctx, z, nz=1):
        batch_size = ctx.size(0)

        ## enc
        #ctx = self.enc(x)

        # view
        assert z.size(0) == batch_size*nz
        ctx = ctx.unsqueeze(1).expand(-1, nz, -1).contiguous()
        ctx = ctx.view(batch_size*nz, -1)

        # concat
        ctxz = torch.cat([ctx, z], dim=1)

        # forward
        h = self.fc(ctxz)
        mu, logvar = self.reparam(h)

        # sample
        z0 = self.sample(mu, logvar)

        return z0, mu, logvar

class VAE(nn.Module):
    def __init__(self,
                 energy_func=normal_energy_func,
                 input_height=28,
                 input_channels=1,
                 z0_dim=100,
                 z_dim=32,
                 c_dim=450,
                 nonlinearity='elu',
                 do_center=False,
                 ):
        super().__init__()
        self.energy_func = energy_func
        self.input_height = input_height
        self.input_channels = input_channels
        self.z0_dim = z0_dim
        self.z_dim = z_dim
        self.latent_dim = z_dim # for ais
        self.c_dim = c_dim
        self.nonlinearity = nonlinearity
        self.do_center = do_center

        assert input_height == 28
        assert input_channels == 1
        assert nonlinearity == 'elu'

        self.inp_encode = InputEncoder(c_dim=c_dim, act=nn.ELU(), do_center=do_center)
        self.aux_encode = AuxEncoder(z0_dim=z0_dim, c_dim=c_dim, act=nn.ELU(), do_center=do_center)
        self.encode = Encoder(z0_dim=z0_dim, z_dim=z_dim, c_dim=c_dim, act=nn.ELU(), do_center=do_center)
        self.decode = Decoder(z_dim=z_dim, c_dim=c_dim, act=nn.ELU())
        self.aux_decode = AuxDecoder(z_dim=z_dim, z0_dim=z0_dim, c_dim=c_dim, act=nn.ELU(), do_center=do_center)

    def loss(self,
             mu_qz, logvar_qz,
             mu_qz0, logvar_qz0,
             mu_pz0, logvar_pz0,
             logit_px, target_x,
             beta=1.0,
             ):
        # kld loss: log q(z|z0, x) - log p(z)
        kld_loss = loss_kld_gaussian(mu_qz, logvar_qz, do_sum=False)

        # aux dec loss: -log r(z0|z,x)
        aux_kld_loss = loss_kld_gaussian_vs_gaussian(
                mu_qz0, logvar_qz0,
                mu_pz0, logvar_pz0,
                do_sum=False,
                )

        # recon loss (neg likelihood): -log p(x|z)
        recon_loss = loss_recon_bernoulli_with_logit(logit_px, target_x, do_sum=False)

        # add loss
        loss = recon_loss + beta*kld_loss + beta*aux_kld_loss
        return loss.mean(), recon_loss.mean(), kld_loss.mean(), aux_kld_loss.mean()

    def forward(self, input, beta=1.0):
        # init
        batch_size = input.size(0)
        input = input.view(batch_size, self.input_channels, self.input_height, self.input_height)

        # inp encode
        inp = self.inp_encode(input)

        # aux encode
        z0, mu_qz0, logvar_qz0 = self.aux_encode(inp)

        # encode
        z, mu_qz, logvar_qz, _ = self.encode(inp, z0)

        # aux decode
        _, mu_pz0, logvar_pz0 = self.aux_decode(inp, z)

        # decode
        x, logit_px = self.decode(z)

        ''' get loss '''
        loss, recon_loss, kld_loss, aux_kld_loss = self.loss(
                mu_qz, logvar_qz,
                mu_qz0, logvar_qz0,
                mu_pz0, logvar_pz0,
                logit_px, input,
                beta=beta,
                )

        # return
        return x, torch.sigmoid(logit_px), z, loss, recon_loss.detach(), kld_loss.detach()+aux_kld_loss.detach()

    def generate(self, batch_size=1):
        # init mu_z and logvar_z (as unit normal dist)
        weight = next(self.parameters())
        mu_z = weight.new_zeros(batch_size, self.z_dim)
        logvar_z = weight.new_zeros(batch_size, self.z_dim)

        # sample z (from unit normal dist)
        z = sample_gaussian(mu_z, logvar_z) # sample z

        # decode
        output, logit_px = self.decode(z)

        # return
        return output, torch.sigmoid(logit_px), z

    def logprob(self, input, sample_size=128, z=None):
        #assert int(math.sqrt(sample_size))**2 == sample_size
        # init
        batch_size = input.size(0)
        sample_size1 = sample_size #int(math.sqrt(sample_size))
        sample_size2 = 1 #int(math.sqrt(sample_size))
        input = input.view(batch_size, self.input_channels, self.input_height, self.input_height)

        ''' encode input ''' 
        inp = self.inp_encode(input)

        ''' get - (log q(z|z0,x) + log q(z0|z) - log p(z0|z,x) - log p(z)) '''
        ''' get log q(z0|x) '''
        _, mu_qz0, logvar_qz0 = self.aux_encode(inp)
        mu_qz0 = mu_qz0.unsqueeze(1).expand(batch_size, sample_size1, self.z0_dim).contiguous().view(batch_size*sample_size1, self.z0_dim) # bsz*ssz1 x z0_dim
        logvar_qz0 = logvar_qz0.unsqueeze(1).expand(batch_size, sample_size1, self.z0_dim).contiguous().view(batch_size*sample_size1, self.z0_dim) # bsz*ssz1 x z0_dim
        z0 = self.aux_encode.sample(mu_qz0, logvar_qz0) # bsz*ssz1 x z0_dim
        log_qz0 = logprob_gaussian(mu_qz0, logvar_qz0, z0, do_unsqueeze=False, do_mean=False)
        log_qz0 = torch.sum(log_qz0.view(batch_size, sample_size1, self.z0_dim), dim=2) # bsz x ssz1
        log_qz0 = log_qz0.unsqueeze(2).expand(batch_size, sample_size1, sample_size2).contiguous().view(batch_size, sample_size1*sample_size2) # bsz x ssz1*ssz2

        ''' get log q(z|z0,x) '''
        # forward
        _, mu_qz, logvar_qz, _ = self.encode(inp, z0, nz=sample_size1) # bsz*ssz1 x z_dim
        mu_qz = mu_qz.detach().repeat(1, sample_size2).view(batch_size*sample_size1, sample_size2, self.z_dim)
        logvar_qz = logvar_qz.detach().repeat(1, sample_size2).view(batch_size*sample_size1, sample_size2, self.z_dim)
        z = self.encode.sample(mu_qz, logvar_qz) # bsz x ssz1 x ssz2 x z_dim
        log_qz = logprob_gaussian(mu_qz, logvar_qz, z, do_unsqueeze=False, do_mean=False)
        log_qz = torch.sum(log_qz.view(batch_size, sample_size1*sample_size2, self.z_dim), dim=2) # bsz x ssz1*ssz2

        ''' get log p(z0|z,x) '''
        # encode
        _z0 = z0.unsqueeze(1).expand(batch_size*sample_size1, sample_size2, self.z0_dim).contiguous().view(batch_size, sample_size1, sample_size2, self.z0_dim).detach()
        _, mu_pz0, logvar_pz0 = self.aux_decode(inp, z.view(-1, self.z_dim), nz=sample_size1*sample_size2) # bsz*ssz1 x z_dim
        mu_pz0 = mu_pz0.view(batch_size, sample_size1, sample_size2, self.z0_dim)
        logvar_pz0 = logvar_pz0.view(batch_size, sample_size1, sample_size2, self.z0_dim)
        log_pz0 = logprob_gaussian(mu_pz0, logvar_pz0, _z0, do_unsqueeze=False, do_mean=False) # bsz x ssz1 x ssz2 xz0_dim
        log_pz0 = torch.sum(log_pz0.view(batch_size, sample_size1*sample_size2, self.z0_dim), dim=2) # bsz x ssz1*ssz2

        ''' get log p(z) '''
        # get prior (as unit normal dist)
        mu_pz = input.new_zeros(batch_size*sample_size1, sample_size2, self.z_dim)
        logvar_pz = input.new_zeros(batch_size*sample_size1, sample_size2, self.z_dim)
        log_pz = logprob_gaussian(mu_pz, logvar_pz, z, do_unsqueeze=False, do_mean=False)
        log_pz = torch.sum(log_pz.view(batch_size, sample_size1*sample_size2, self.z_dim), dim=2) # bsz x ssz1*ssz2

        ''' get log p(x|z) '''
        # decode
        _input = input.unsqueeze(1).unsqueeze(1).expand(
                batch_size, sample_size1, sample_size2, self.input_channels, self.input_height, self.input_height) # bsz x ssz1 x ssz2 x input_dim
        _z = z.view(-1, self.z_dim)
        #_, mu_x, logvar_x = self.decode(_z) # bsz*ssz1*ssz2 x zdim
        #mu_x = mu_x.view(batch_size, sample_size1, sample_size2, self.input_dim)
        #logvar_x = logvar_x.view(batch_size, sample_size1, sample_size2, self.input_dim)
        #loglikelihood = logprob_gaussian(mu_x, logvar_x, _input, do_unsqueeze=False, do_mean=False)
        _, logit_px = self.decode(_z) # bsz*ssz1*ssz2 x zdim
        logit_px = logit_px.view(batch_size, sample_size1, sample_size2, self.input_channels, self.input_height, self.input_height)
        loglikelihood = -F.binary_cross_entropy_with_logits(logit_px, _input, reduction='none')
        loglikelihood = torch.sum(loglikelihood.view(batch_size, sample_size1*sample_size2, -1), dim=2) # bsz x ssz1*ssz2

        ''' get log p(x|z)p(z)/q(z|x) '''
        logprob = loglikelihood + log_pz + log_pz0 - log_qz - log_qz0 # bsz x ssz1*ssz2
        logprob_max, _ = torch.max(logprob, dim=1, keepdim=True)
        rprob = (logprob - logprob_max).exp() # relative prob
        logprob = torch.log(torch.mean(rprob, dim=1, keepdim=True) + 1e-10) + logprob_max # bsz x 1

        # return
        return logprob.mean()
