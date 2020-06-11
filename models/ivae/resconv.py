import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.distributions import MultivariateNormal

from models.layers import Identity, MLP, ResMLP
from models.reparam import NormalDistributionLinear
from utils import loss_kld_gaussian, loss_kld_gaussian_vs_gaussian, loss_recon_gaussian, loss_recon_bernoulli_with_logit, normal_energy_func
from utils import logprob_gaussian, get_covmat
from utils import get_nonlinear_func
from utils import conv_out_size, deconv_out_size
import models.layers2 as nn_

from models.vae.resconv import Decoder

from utils import expand_tensor
from utils import cond_jac_clamping_loss

def get_afunc(nonlinearity_type='elu'):
    if nonlinearity_type == 'relu':
        return nn.ReLU()
    elif nonlinearity_type == 'elu':
        return nn.ELU()
    elif nonlinearity_type == 'softplus':
        return nn.Softplus()
    else:
        raise NotImplementedError

def sample_noise(sz, std=None, device=torch.device('cpu')):
    std = std if std is not None else 1
    eps = torch.randn(*sz).to(device)
    return std * eps

def sample_gaussian(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + std * eps

def convert_2d_3d_tensor(input, sample_size):
    assert input.dim() == 2
    input_expanded, _ = expand_tensor(input, sample_size, do_unsqueeze=True)
    return input_expanded

def convert_4d_5d_tensor(input, sample_size):
    assert input.dim() == 4
    input_expanded, _ = expand_tensor(input, sample_size, do_unsqueeze=True)
    return input_expanded

class Encoder(nn.Module):
    def __init__(self,
                 noise_dim=100,
                 z_dim=32,
                 c_dim=512,#450,
                 h_dim=800,
                 num_hidden_layers=1,
                 nonlinearity='elu', #act=nn.ELU(),
                 do_center=False,
                 enc_noise=False,
                 enc_type='mlp',
                 ):
        super().__init__()
        self.noise_dim = noise_dim
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.h_dim = h_dim
        self.num_hidden_layers = num_hidden_layers
        assert num_hidden_layers > 0
        self.nonlinearity = nonlinearity
        self.do_center = do_center
        self.enc_noise = enc_noise
        nos_dim = noise_dim if not enc_noise else c_dim
        self.enc_type = enc_type
        assert enc_type in ['mlp', 'res-wn-mlp', 'res-mlp', 'res-wn-mlp-lin', 'res-mlp-lin']

        act = get_afunc(nonlinearity_type=nonlinearity)

        self.inp_encode = nn.Sequential(
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
        #self.fc = nn.Sequential(
        #    nn.Linear(c_dim + nos_dim, h_dim, bias=True),
        #    act,
        #    nn.Linear(h_dim, z_dim, bias=True),
        #    )
        if enc_type == 'mlp':
            self.fc = MLP(input_dim=c_dim+nos_dim, hidden_dim=h_dim, output_dim=z_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers, use_nonlinearity_output=False)
        elif enc_type == 'res-wn-mlp':
            self.fc = ResMLP(input_dim=c_dim+nos_dim, hidden_dim=h_dim, output_dim=z_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers, use_nonlinearity_output=False, layer='wnlinear')
        elif enc_type == 'res-mlp':
            self.fc = ResMLP(input_dim=c_dim+nos_dim, hidden_dim=h_dim, output_dim=z_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers, use_nonlinearity_output=False, layer='linear')
        elif enc_type == 'res-wn-mlp-lin':
            self.fc = nn.Sequential(
                    ResMLP(input_dim=c_dim+nos_dim, hidden_dim=h_dim, output_dim=h_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers-1, use_nonlinearity_output=True, layer='wnlinear'),
                    nn.Linear(h_dim, z_dim, bias=True),
                    )
        elif enc_type == 'res-mlp-lin':
            self.fc = nn.Sequential(
                    ResMLP(input_dim=c_dim+nos_dim, hidden_dim=h_dim, output_dim=h_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers-1, use_nonlinearity_output=True, layer='linear'),
                    nn.Linear(h_dim, z_dim, bias=True),
                    )
        else:
            raise NotImplementedError

        self.nos_encode = Identity() if not enc_noise \
                else nn.Sequential(
                    nn.Linear(noise_dim, c_dim, bias=True),
                    act,
                    )

    def sample_noise(self, batch_size, std=None, device=None):
        return sample_noise(sz=[batch_size, self.noise_dim], std=std, device=device)

    def _forward_inp(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 28, 28)

        # rescale
        if self.do_center:
            x = 2*x -1

        # enc
        inp = self.inp_encode(x)

        return inp

    def _forward_nos(self, batch_size=None, noise=None, std=None, device=None):
        assert batch_size is not None or noise is not None
        if noise is None:
            noise = self.sample_noise(batch_size, std=std, device=device)

        # enc
        nos = self.nos_encode(noise)

        return nos

    def _forward_all(self, inp, nos):
        # concat
        inp_nos = torch.cat([inp, nos], dim=1)

        # forward
        z = self.fc(inp_nos)

        return z

    def forward(self, x, noise=None, std=None, nz=1):
        batch_size = x.size(0)
        if noise is None:
            noise = self.sample_noise(batch_size*nz, std=std, device=x.device)
        else:
            assert noise.size(0) == batch_size*nz
            assert noise.size(1) == self.noise_dim

        # enc
        nos = self._forward_nos(noise=noise, std=std, device=x.device)
        inp = self._forward_inp(x)

        # view
        inp = inp.unsqueeze(1).expand(-1, nz, -1).contiguous()
        inp = inp.view(batch_size*nz, -1)

        # forward
        z = self._forward_all(inp, nos)

        return z.view(batch_size, nz, -1)

class ImplicitPosteriorVAE(nn.Module):
    def __init__(self,
                 energy_func=normal_energy_func,
                 input_height=28,
                 input_channels=1,
                 z_dim=32,
                 noise_dim=100,
                 c_dim=512, #450,
                 h_dim=800,
                 num_hidden_layers=1,
                 nonlinearity='elu',
                 do_center=False,
                 do_m5bias=False,
                 enc_noise=False,
                 enc_type='mlp',
                 ):
        super().__init__()
        self.energy_func = energy_func
        self.input_height = input_height
        self.input_channels = input_channels
        self.z_dim = z_dim
        self.latent_dim = z_dim # for ais
        self.noise_dim = noise_dim
        self.c_dim = c_dim
        self.h_dim = h_dim
        self.num_hidden_layers = num_hidden_layers
        self.nonlinearity = nonlinearity
        self.do_center = do_center
        self.do_m5bias = do_m5bias
        self.enc_noise = enc_noise
        self.enc_type = enc_type

        assert input_height == 28
        assert input_channels == 1
        #if nonlinearity == 'elu':
        #    afunc = nn.ELU()
        #elif nonlinearity == 'softplus':
        #    afunc = nn.Softplus()
        #else:
        #    raise NotImplementedError

        self.encode = Encoder(noise_dim=noise_dim, z_dim=z_dim, c_dim=c_dim, h_dim=h_dim, num_hidden_layers=num_hidden_layers, nonlinearity=nonlinearity, do_center=do_center, enc_noise=enc_noise, enc_type=enc_type)
        self.decode = Decoder(z_dim=z_dim, c_dim=c_dim, act=nn.ELU(), do_m5bias=do_m5bias)

    def loss(self, z, logit_x, target_x, beta=1.0):
        # loss from energy func
        prior_loss = self.energy_func(z.view(-1, self.z_dim))

        # recon loss (neg likelihood): -log p(x|z)
        recon_loss = loss_recon_bernoulli_with_logit(logit_x, target_x, do_sum=False)

        # add loss
        loss = recon_loss + beta*prior_loss
        return loss.mean(), recon_loss.mean(), prior_loss.mean()

    def jac_clamping_loss(self, input, z, eps, std, nz, eta_min, p=2, EPS=1.):
        raise NotImplementedError

    def forward_hidden(self, input, std=None, nz=1):
        # init
        batch_size = input.size(0)
        input = input.view(batch_size, self.input_channels, self.input_height, self.input_height)

        # gen noise source
        eps = sample_noise(sz=[batch_size*nz, self.noise_dim], std=std, device=input.device)

        # sample z
        z = self.encode(input, noise=eps, std=std, nz=nz)

        return z

    def forward(self, input, beta=1.0, eta=0.0, lmbd=0.0, std=None, nz=1):
        # init
        batch_size = input.size(0)
        input = input.view(batch_size, self.input_channels, self.input_height, self.input_height)
        input_expanded = convert_4d_5d_tensor(input, sample_size=nz)
        input_expanded_flattened = input_expanded.view(batch_size*nz, self.input_channels, self.input_height, self.input_height)

        # gen noise source
        eps = sample_noise(sz=[batch_size*nz, self.noise_dim], std=std, device=input.device)

        # sample z
        z = self.encode(input, noise=eps, std=std, nz=nz)

        # z flattten
        z_flattened = z.view(batch_size*nz, -1)

        # decode
        x, logit_x = self.decode(z_flattened)

        # loss
        if lmbd > 0:
            raise NotImplementedError
            jaclmp_loss = lmbd*self.jac_clamping_loss(input, z, eps, std=std, nz=nz, eta_min=eta)
        else:
            jaclmp_loss = 0
        loss, recon_loss, prior_loss = self.loss(
                z_flattened,
                logit_x, input_expanded_flattened,
                beta=beta,
                )
        loss += jaclmp_loss

        # return
        return x, torch.sigmoid(logit_x), z, loss, recon_loss.detach(), prior_loss.detach()

    def generate(self, batch_size=1):
        # init mu_z and logvar_z (as unit normal dist)
        weight = next(self.parameters())
        mu_z = weight.new_zeros(batch_size, self.z_dim)
        logvar_z = weight.new_zeros(batch_size, self.z_dim)

        # sample z (from unit normal dist)
        z = sample_gaussian(mu_z, logvar_z) # sample z

        # decode
        output, logit_x = self.decode(z)

        # return
        return output, torch.sigmoid(logit_x), z

    def logprob(self, input, sample_size=128, z=None, std=None):
        return self.logprob_w_cov_gaussian_posterior(input, sample_size, z, std)

    def logprob_w_cov_gaussian_posterior(self, input, sample_size=128, z=None, std=None):
        # init
        batch_size = input.size(0)
        input = input.view(batch_size, self.input_channels, self.input_height, self.input_height)
        assert sample_size >= 2*self.z_dim

        ''' get z and pseudo log q(newz|x) '''
        z, newz = [], []
        logposterior = []
        inp = self.encode._forward_inp(input).detach()
        for i in range(batch_size):
            _inp = inp[i:i+1, :].expand(sample_size, inp.size(1))
            _nos = self.encode._forward_nos(batch_size=sample_size, std=std, device=input.device).detach()
            _z = self.encode._forward_all(_inp, _nos) # ssz x zdim
            z += [_z.detach().unsqueeze(0)]
        z = torch.cat(z, dim=0) # bsz x ssz x zdim
        mu_qz = torch.mean(z, dim=1) # bsz x zdim
        for i in range(batch_size):
            _cov_qz = get_covmat(z[i, :, :])
            _rv_z = MultivariateNormal(mu_qz[i], _cov_qz)
            _newz = _rv_z.rsample(torch.Size([1, sample_size]))
            _logposterior = _rv_z.log_prob(_newz)

            newz += [_newz]
            logposterior += [_logposterior]
        newz = torch.cat(newz, dim=0) # bsz x ssz x zdim
        logposterior = torch.cat(logposterior, dim=0) # bsz x ssz

        ''' get log p(z) '''
        # get prior (as unit normal dist)
        mu_pz = input.new_zeros(batch_size, sample_size, self.z_dim)
        logvar_pz = input.new_zeros(batch_size, sample_size, self.z_dim)
        logprior = logprob_gaussian(mu_pz, logvar_pz, newz, do_unsqueeze=False, do_mean=False)
        logprior = torch.sum(logprior.view(batch_size, sample_size, self.z_dim), dim=2) # bsz x ssz

        ''' get log p(x|z) '''
        # decode
        logit_x = []
        #for i in range(sample_size):
        for i in range(batch_size):
            _, _logit_x = self.decode(newz[i, :, :]) # ssz x zdim
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
