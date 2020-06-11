import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from models.layers import MLP
from models.reparam import NormalDistributionLinear
from utils import loss_kld_gaussian, loss_recon_gaussian, normal_energy_func
from utils import logprob_gaussian
from utils import get_nonlinear_func


def sample_gaussian(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + std * eps

class Encoder(nn.Module):
    def __init__(self,
                 input_dim=2,
                 h_dim=64,
                 z_dim=2,
                 nonlinearity='softplus',
                 num_hidden_layers=1,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.nonlinearity = nonlinearity
        self.num_hidden_layers = num_hidden_layers

        self.main = MLP(input_dim=input_dim, hidden_dim=h_dim, output_dim=h_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers-1, use_nonlinearity_output=True)
        self.reparam = NormalDistributionLinear(h_dim, z_dim)

    def sample(self, mu, logvar):
        return self.reparam.sample_gaussian(mu, logvar)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.input_dim)

        # forward
        h = self.main(x) 
        mu, logvar = self.reparam(h)

        # sample
        z = self.sample(mu, logvar)

        return z, mu, logvar

class Decoder(nn.Module):
    def __init__(self,
                 input_dim=2,
                 h_dim=64,
                 z_dim=2,
                 nonlinearity='tanh',
                 num_hidden_layers=1,
                 init='gaussian', #None,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.nonlinearity = nonlinearity
        self.num_hidden_layers = num_hidden_layers
        self.init = init

        self.main = MLP(input_dim=z_dim, hidden_dim=h_dim, output_dim=h_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers-1, use_nonlinearity_output=True)
        self.reparam = NormalDistributionLinear(h_dim, input_dim)

        if self.init == 'gaussian':
            self.reset_parameters()
        else:
            pass

    def reset_parameters(self):
        nn.init.normal_(self.reparam.mean_fn.weight)

    def sample(self, mu, logvar):
        return self.reparam.sample_gaussian(mu, logvar)

    def forward(self, z):
        batch_size = z.size(0)
        z = z.view(batch_size, -1)

        # forward
        h = self.main(z)
        mu, logvar = self.reparam(h)

        # sample
        x = self.sample(mu, logvar)

        return x, mu, logvar

class VAE(nn.Module):
    def __init__(self,
                 energy_func=normal_energy_func,
                 input_dim=2,
                 h_dim=64,
                 z_dim=2,
                 nonlinearity='softplus',
                 num_hidden_layers=1,
                 init='gaussian', #None,
                 ):
        super().__init__()
        self.energy_func = energy_func
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.latent_dim = self.z_dim # for ais
        self.nonlinearity = nonlinearity
        self.num_hidden_layers = num_hidden_layers
        self.init = init

        self.encode = Encoder(input_dim, h_dim, z_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers)
        self.decode = Decoder(input_dim, h_dim, z_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers)

    def loss(self, mu_z, logvar_z, mu_x, logvar_x, target_x, beta=1.0):
        # kld loss
        kld_loss = loss_kld_gaussian(mu_z, logvar_z, do_sum=False)

        # recon loss (likelihood)
        recon_loss = loss_recon_gaussian(mu_x, logvar_x, target_x.view(-1, 2), do_sum=False)

        # add loss
        loss = recon_loss + beta*kld_loss
        return loss.mean(), recon_loss.mean(), kld_loss.mean()

    def forward(self, input, beta=1.0):
        # init
        batch_size = input.size(0)
        input = input.view(batch_size, self.input_dim)

        # encode
        z, mu_z, logvar_z = self.encode(input)

        # decode
        x, mu_x, logvar_x = self.decode(z)

        # loss
        loss, recon_loss, kld_loss \
                = self.loss(mu_z, logvar_z,
                            mu_x, logvar_x, input,
                            beta=beta,
                            )

        # return
        return x, mu_x, z, loss, recon_loss.detach(), kld_loss.detach()

    def generate(self, batch_size=1):
        # init mu_z and logvar_z (as unit normal dist)
        weight = next(self.parameters())
        mu_z = weight.new_zeros(batch_size, self.z_dim)
        logvar_z = weight.new_zeros(batch_size, self.z_dim)

        # sample z (from unit normal dist)
        z = sample_gaussian(mu_z, logvar_z)  # sample z

        # decode
        output, mu_x, logvar_x = self.decode(z)

        # return
        return output, mu_x, z

    def logprob(self, input, sample_size=128, z=None):
        # init
        batch_size = input.size(0)
        input = input.view(batch_size, self.input_dim)

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
        #mu_x, logvar_x = [], []
        #for i in range(batch_size):
        #    _, _mu_x, _logvar_x = self.decode(z[i, :, :]) # ssz x zdim
        #    mu_x += [_mu_x.detach().unsqueeze(0)]
        #    logvar_x += [_logvar_x.detach().unsqueeze(0)]
        #mu_x = torch.cat(mu_x, dim=0) # bsz x ssz x input_dim
        #logvar_x = torch.cat(logvar_x, dim=0) # bsz x ssz x input_dim
        _z = z.view(-1, self.z_dim)
        _, mu_x, logvar_x = self.decode(_z) # bsz*ssz x zdim
        mu_x = mu_x.view(batch_size, sample_size, self.input_dim)
        logvar_x = logvar_x.view(batch_size, sample_size, self.input_dim)
        _input = input.unsqueeze(1).expand(batch_size, sample_size, self.input_dim) # bsz x ssz x input_dim
        loglikelihood = logprob_gaussian(mu_x, logvar_x, _input, do_unsqueeze=False, do_mean=False)
        loglikelihood = torch.sum(loglikelihood, dim=2) # bsz x ssz

        ''' get log p(x|z)p(z)/q(z|x) '''
        logprob = loglikelihood + logprior - logposterior # bsz x ssz
        logprob_max, _ = torch.max(logprob, dim=1, keepdim=True)
        rprob = (logprob - logprob_max).exp() # relative prob
        logprob = torch.log(torch.mean(rprob, dim=1, keepdim=True) + 1e-10) + logprob_max # bsz x 1

        # return
        return logprob.mean()

    def logprob_w_prior(self, input, sample_size=128, z=None):
        # init
        batch_size = input.size(0)
        input = input.view(batch_size, self.input_dim)

        ''' get z samples from p(z) '''
        # get prior (as unit normal dist)
        if z is None:
            mu_pz = input.new_zeros(batch_size, sample_size, self.z_dim)
            logvar_pz = input.new_zeros(batch_size, sample_size, self.z_dim)
            z = sample_gaussian(mu_pz, logvar_pz)  # sample z

        ''' get log p(x|z) '''
        # decode
        _z = z.view(-1, self.z_dim)
        _, mu_x, logvar_x = self.decode(_z) # bsz*ssz x zdim
        mu_x = mu_x.view(batch_size, sample_size, self.input_dim)
        logvar_x = logvar_x.view(batch_size, sample_size, self.input_dim)
        _input = input.unsqueeze(1).expand(batch_size, sample_size, self.input_dim) # bsz x ssz x input_dim
        loglikelihood = logprob_gaussian(mu_x, logvar_x, _input, do_unsqueeze=False, do_mean=False)
        loglikelihood = torch.sum(loglikelihood, dim=2) # bsz x ssz

        ''' get log p(x) '''
        logprob = loglikelihood # bsz x ssz
        logprob_max, _ = torch.max(logprob, dim=1, keepdim=True)
        rprob = (logprob-logprob_max).exp() # relative prob
        logprob = torch.log(torch.mean(rprob, dim=1, keepdim=True) + 1e-10) + logprob_max # bsz x 1

        # return
        return logprob.mean()
