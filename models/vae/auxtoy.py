import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from models.layers import Identity, MLP
from models.reparam import NormalDistributionLinear
from utils import loss_kld_gaussian, loss_kld_gaussian_vs_gaussian, loss_recon_gaussian, normal_energy_func
from utils import logprob_gaussian
from utils import get_nonlinear_func
from models.vae.toy import Decoder


def sample_gaussian(mu, logvar, _std=1.):
    if _std is None:
        _std = 1.
    std = _std*torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + std * eps

class AuxEncoder(nn.Module):
    def __init__(self,
                 input_dim=2,
                 h_dim=8,
                 noise_dim=2,
                 nonlinearity='softplus',
                 num_hidden_layers=1,
                 clip_logvar=None,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.noise_dim = noise_dim
        self.nonlinearity = nonlinearity
        self.num_hidden_layers = num_hidden_layers
        self.clip_logvar = clip_logvar

        self.main = MLP(input_dim=input_dim, hidden_dim=h_dim, output_dim=h_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers-1, use_nonlinearity_output=True)
        self.reparam = NormalDistributionLinear(h_dim, noise_dim, nonlinearity=clip_logvar)

    def sample(self, mu, logvar, _std=1.):
        return sample_gaussian(mu, logvar, _std=_std)

    def forward(self, x, _std=1.):
        batch_size = x.size(0)
        x = x.view(batch_size, self.input_dim)

        # forward
        h = self.main(x)
        mu, logvar = self.reparam(h)

        # sample
        noise = self.sample(mu, logvar, _std=_std)

        return noise, mu, logvar, h

class Encoder(nn.Module):
    def __init__(self,
                 input_dim=2,
                 noise_dim=2,
                 h_dim=8,
                 z_dim=2,
                 nonlinearity='softplus',
                 num_hidden_layers=1,
                 enc_input=False,
                 enc_noise=False,
                 clip_logvar=None,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.noise_dim = noise_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.nonlinearity = nonlinearity
        self.num_hidden_layers = num_hidden_layers
        self.enc_input = enc_input
        self.enc_noise = enc_noise
        self.clip_logvar = clip_logvar
        #self.inp_dim = input_dim if not enc_input else h_dim
        #self.ctx_dim = noise_dim if not enc_noise else h_dim

        self.inp_encode = None
        self.nos_encode = None
        self.fc = None
        self.reparam = None

        #self.main = MLP(input_dim=input_dim, hidden_dim=h_dim, output_dim=h_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers, use_nonlinearity_output=True)
        #self.reparam = NormalDistributionLinear(h_dim, z_dim, nonlinearity=clip_logvar)

    def sample(self, mu_z, logvar_z):
        raise NotImplementedError
        #return self.reparam.sample_gaussian(mu_z, logvar_z)

    def _forward_inp(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.input_dim)

        # enc
        inp = self.inp_encode(x)

        return inp

    def _forward_nos(self, noise):
        # enc
        nos = self.nos_encode(noise)

        return nos

    def _forward_all(self, inp, nos):
        raise NotImplementedError
        #hid = self.fc(inp, nos)
        #mu_z, logvar_ = self.reparam(hid)
        #z = self.sample(mu_z, logvar_z)
        return z, mu_z, logvar_z, h

    def forward(self, x, noise, nz=1):
        batch_size = x.size(0)

        # enc
        nos = self._forward_nos(noise)
        inp = self._forward_inp(x)

        # view
        assert nos.size(0) == batch_size*nz
        inp = inp.unsqueeze(1).expand(-1, nz, -1).contiguous()
        inp = inp.view(batch_size*nz, -1)

        # forward
        z, mu_z, logvar_z, h = self._forward_all(inp, nos)
        #z, mu_z, logvar_z = self._forward_all(inp, nos)

        return z, mu_z, logvar_z, h
        #return z, mu_z, logvar_z

class SimpleEncoder(Encoder):
    def __init__(self,
                 input_dim=2,
                 noise_dim=2,
                 h_dim=64,
                 z_dim=2,
                 nonlinearity='tanh',
                 num_hidden_layers=1,
                 enc_input=False,
                 enc_noise=False,
                 clip_logvar=None,
                 ):
        super().__init__(
        input_dim = input_dim,
        noise_dim = noise_dim,
        h_dim = h_dim,
        z_dim = z_dim,
        nonlinearity = nonlinearity,
        num_hidden_layers = num_hidden_layers,
        enc_input = enc_input,
        enc_noise = enc_noise,
        clip_logvar = clip_logvar,
        )
        inp_dim = input_dim if not enc_input else h_dim
        ctx_dim = noise_dim if not enc_noise else h_dim

        self.inp_encode = Identity() if not enc_input \
                else MLP(input_dim=input_dim, hidden_dim=h_dim, output_dim=h_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers-1, use_nonlinearity_output=True)
        self.nos_encode = Identity() if not enc_noise \
                else MLP(input_dim=noise_dim, hidden_dim=h_dim, output_dim=h_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers-1, use_nonlinearity_output=True)
        self.fc = MLP(input_dim=inp_dim+ctx_dim, hidden_dim=h_dim, output_dim=h_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers-1, use_nonlinearity_output=True)
        self.reparam = NormalDistributionLinear(h_dim, z_dim, nonlinearity=clip_logvar)

    def sample(self, mu_z, logvar_z):
        return self.reparam.sample_gaussian(mu_z, logvar_z)

    def _forward_all(self, inp, nos):
        h1 = torch.cat([inp, nos], dim=1)
        h2 = self.fc(h1)
        mu_z, logvar_z = self.reparam(h2)
        z = self.sample(mu_z, logvar_z)
        return z, mu_z, logvar_z, h2

class AuxDecoder(nn.Module):
    def __init__(self,
                 input_dim=2,
                 z_dim=2,
                 noise_dim=2,
                 h_dim=64,
                 nonlinearity='tanh',
                 num_hidden_layers=1,
                 enc_input=False,
                 enc_latent=False,
                 clip_logvar=None,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.noise_dim = noise_dim
        self.h_dim = h_dim
        self.nonlinearity = nonlinearity
        self.num_hidden_layers = num_hidden_layers
        self.enc_input = enc_input
        self.enc_latent = enc_latent
        inp_dim = input_dim if not enc_input else h_dim
        ltt_dim = z_dim if not enc_latent else h_dim

        self.inp_encode = Identity() if not enc_input \
                else MLP(input_dim=input_dim, hidden_dim=h_dim, output_dim=h_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers-1, use_nonlinearity_output=True)
        self.ltt_encode = Identity() if not enc_latent \
                else MLP(input_dim=z_dim, hidden_dim=h_dim, output_dim=h_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers-1, use_nonlinearity_output=True)
        self.fc = MLP(input_dim=inp_dim+ltt_dim, hidden_dim=h_dim, output_dim=h_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers-1, use_nonlinearity_output=True)
        self.reparam = NormalDistributionLinear(h_dim, noise_dim, nonlinearity=clip_logvar)

    def sample(self, mu, logvar):
        return self.reparam.sample_gaussian(mu, logvar)

    def _forward_inp(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.input_dim)

        # enc
        inp = self.inp_encode(x)

        return inp

    def _forward_ltt(self, z):
        # enc
        ltt = self.ltt_encode(z)

        return ltt

    def _forward_all(self, inp, ltt):
        h1 = torch.cat([inp, ltt], dim=1)
        h2 = self.fc(h1)
        mu_n, logvar_n = self.reparam(h2)
        noise = self.sample(mu_n, logvar_n)
        return noise, mu_n, logvar_n

    def forward(self, x, z, nz=1):
        batch_size = x.size(0)

        # enc
        ltt = self._forward_ltt(z)
        inp = self._forward_inp(x)

        # view
        assert ltt.size(0) == batch_size*nz
        inp = inp.unsqueeze(1).expand(-1, nz, -1).contiguous()
        inp = inp.view(batch_size*nz, -1)

        # forward
        noise, mu_n, logvar_n = self._forward_all(inp, ltt)

        return noise, mu_n, logvar_n

class VAE(nn.Module):
    def __init__(self,
                 energy_func=normal_energy_func,
                 input_dim=2,
                 noise_dim=2,
                 h_dim=64,
                 z_dim=2,
                 nonlinearity='tanh',
                 num_hidden_layers=1,
                 init='gaussian', #None,
                 enc_type='simple',
                 clip_logvar=None,
                 ):
        super().__init__()
        self.energy_func = energy_func
        self.input_dim = input_dim
        self.noise_dim = noise_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.latent_dim = z_dim # for ais
        self.nonlinearity = nonlinearity
        self.num_hidden_layers = num_hidden_layers
        self.init = init
        clip_logvar = None if clip_logvar == 'none' else clip_logvar
        self.clip_logvar = clip_logvar
        self.enc_type = enc_type
        assert enc_type in ['simple']

        self.aux_encode = AuxEncoder(input_dim, h_dim, noise_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers, clip_logvar=clip_logvar)
        if enc_type == 'simple':
            self.encode = SimpleEncoder(input_dim, noise_dim, h_dim, z_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers, clip_logvar=None)
        else:
            raise NotImplementedError
        self.decode = Decoder(input_dim, h_dim, z_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers, init=init)
        self.aux_decode = AuxDecoder(input_dim, z_dim, noise_dim, h_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers)

    def loss(self,
             mu_qz, logvar_qz,
             mu_qz0, logvar_qz0,
             mu_pz0, logvar_pz0,
             mu_x, logvar_x, target_x,
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
        recon_loss = loss_recon_gaussian(mu_x, logvar_x, target_x.view(-1, 2), do_sum=False)

        # add loss
        loss = recon_loss + beta*kld_loss + beta*aux_kld_loss
        return loss.mean(), recon_loss.mean(), kld_loss.mean(), aux_kld_loss.mean()

    def forward(self, input, beta=1.0):
        # init
        batch_size = input.size(0)
        input = input.view(batch_size, self.input_dim)

        # aux encode
        z0, mu_qz0, logvar_qz0, _ = self.aux_encode(input)

        # encode
        z, mu_qz, logvar_qz, _ = self.encode(input, z0)
        #z, mu_qz, logvar_qz = self.encode(input, z0)

        # aux decode
        _, mu_pz0, logvar_pz0 = self.aux_decode(input, z)

        # decode
        x, mu_px, logvar_px = self.decode(z)

        ''' get loss '''
        loss, recon_loss, kld_loss, aux_kld_loss = self.loss(
                mu_qz, logvar_qz,
                mu_qz0, logvar_qz0,
                mu_pz0, logvar_pz0,
                mu_px, logvar_px, input,
                beta=beta,
                )

        # return
        #return x, mu_px, z, loss, recon_loss.detach(), kld_loss.detach(), aux_kld_loss.detach()
        return x, mu_px, z, loss, recon_loss.detach(), kld_loss.detach()+aux_kld_loss.detach()

    def generate(self, batch_size=1):
        # init mu_z and logvar_z (as unit normal dist)
        weight = next(self.parameters())
        mu_z = weight.new_zeros(batch_size, self.z_dim)
        logvar_z = weight.new_zeros(batch_size, self.z_dim)

        # sample z (from unit normal dist)
        z = sample_gaussian(mu_z, logvar_z) # sample z

        # decode
        output, mu_x, logvar_x = self.decode(z)

        # return
        return output, mu_x, z

    def logprob(self, input, sample_size=128, z=None):
        #assert int(math.sqrt(sample_size))**2 == sample_size
        # init
        batch_size = input.size(0)
        sample_size1 = sample_size #int(math.sqrt(sample_size))
        sample_size2 = 1 #int(math.sqrt(sample_size))
        input = input.view(batch_size, self.input_dim)

        ''' get - (log q(z|z0,x) + log q(z0|z) - log p(z0|z,x) - log p(z)) '''
        ''' get log q(z0|x) '''
        _, mu_qz0, logvar_qz0, _ = self.aux_encode(input)
        mu_qz0 = mu_qz0.unsqueeze(1).expand(batch_size, sample_size1, self.noise_dim).contiguous().view(batch_size*sample_size1, self.noise_dim) # bsz*ssz1 x noise_dim
        logvar_qz0 = logvar_qz0.unsqueeze(1).expand(batch_size, sample_size1, self.noise_dim).contiguous().view(batch_size*sample_size1, self.noise_dim) # bsz*ssz1 x noise_dim
        z0 = self.aux_encode.sample(mu_qz0, logvar_qz0) # bsz*ssz1 x noise_dim
        log_qz0 = logprob_gaussian(mu_qz0, logvar_qz0, z0, do_unsqueeze=False, do_mean=False)
        log_qz0 = torch.sum(log_qz0.view(batch_size, sample_size1, self.noise_dim), dim=2) # bsz x ssz1
        log_qz0 = log_qz0.unsqueeze(2).expand(batch_size, sample_size1, sample_size2).contiguous().view(batch_size, sample_size1*sample_size2) # bsz x ssz1*ssz2

        ''' get log q(z|z0,x) '''
        # forward
        nos = self.encode._forward_nos(z0) # bsz*ssz1 x noise_dim
        inp = self.encode._forward_inp(input) # bsz x noise_dim
        inp = inp.unsqueeze(1).expand(batch_size, sample_size1, -1).contiguous().view(batch_size*sample_size1, -1)
        _, mu_qz, logvar_qz, _ = self.encode._forward_all(inp, nos) # bsz*ssz1 x z_dim
        mu_qz = mu_qz.detach().repeat(1, sample_size2).view(batch_size*sample_size1, sample_size2, self.z_dim)
        logvar_qz = logvar_qz.detach().repeat(1, sample_size2).view(batch_size*sample_size1, sample_size2, self.z_dim)
        z = self.encode.sample(mu_qz, logvar_qz) # bsz x ssz1 x ssz2 x z_dim
        log_qz = logprob_gaussian(mu_qz, logvar_qz, z, do_unsqueeze=False, do_mean=False)
        log_qz = torch.sum(log_qz.view(batch_size, sample_size1*sample_size2, self.z_dim), dim=2) # bsz x ssz1*ssz2

        ''' get log p(z0|z,x) '''
        # encode
        _z0 = z0.unsqueeze(1).expand(batch_size*sample_size1, sample_size2, self.noise_dim).contiguous().view(batch_size, sample_size1, sample_size2, self.noise_dim)
        ltt = self.aux_decode._forward_ltt(z.view(-1, self.z_dim)) # bsz*ssz1*ssz2 x z_dim
        inp = self.aux_decode._forward_inp(input) # bsz x inp_dim
        inp = inp.unsqueeze(1).unsqueeze(1).expand(batch_size, sample_size1, sample_size2, -1).contiguous().view(batch_size*sample_size1*sample_size2, -1) # bsz*ss1*ssz2 x inp_dim
        _, mu_pz0, logvar_pz0 = self.aux_decode._forward_all(inp, ltt) # bsz*ssz1 x z_dim
        mu_pz0 = mu_pz0.view(batch_size, sample_size1, sample_size2, self.noise_dim)
        logvar_pz0 = logvar_pz0.view(batch_size, sample_size1, sample_size2, self.noise_dim)
        log_pz0 = logprob_gaussian(mu_pz0, logvar_pz0, _z0, do_unsqueeze=False, do_mean=False) # bsz x ssz1 x ssz2 xnoise_dim
        log_pz0 = torch.sum(log_pz0.view(batch_size, sample_size1*sample_size2, self.noise_dim), dim=2) # bsz x ssz1*ssz2

        ''' get log p(z) '''
        # get prior (as unit normal dist)
        mu_pz = input.new_zeros(batch_size*sample_size1, sample_size2, self.z_dim)
        logvar_pz = input.new_zeros(batch_size*sample_size1, sample_size2, self.z_dim)
        log_pz = logprob_gaussian(mu_pz, logvar_pz, z, do_unsqueeze=False, do_mean=False)
        log_pz = torch.sum(log_pz.view(batch_size, sample_size1*sample_size2, self.z_dim), dim=2) # bsz x ssz1*ssz2

        ''' get log p(x|z) '''
        # decode
        _input = input.unsqueeze(1).unsqueeze(1).expand(
                batch_size, sample_size1, sample_size2, self.input_dim) # bsz x ssz1 x ssz2 x input_dim
        _z = z.view(-1, self.z_dim)
        _, mu_x, logvar_x = self.decode(_z) # bsz*ssz1*ssz2 x zdim
        mu_x = mu_x.view(batch_size, sample_size1, sample_size2, self.input_dim)
        logvar_x = logvar_x.view(batch_size, sample_size1, sample_size2, self.input_dim)
        loglikelihood = logprob_gaussian(mu_x, logvar_x, _input, do_unsqueeze=False, do_mean=False)
        #_, logit_x = self.decode(_z) # bsz*ssz1*ssz2 x zdim
        #logit_x = logit_x.view(batch_size, sample_size1, sample_size2, self.input_dim)
        #loglikelihood = -F.binary_cross_entropy_with_logits(logit_x, _input, reduction='none')
        loglikelihood = torch.sum(loglikelihood.view(batch_size, sample_size1*sample_size2, self.input_dim), dim=2) # bsz x ssz1*ssz2

        ''' get log p(x|z)p(z)/q(z|x) '''
        logprob = loglikelihood + log_pz + log_pz0 - log_qz - log_qz0 # bsz x ssz1*ssz2
        logprob_max, _ = torch.max(logprob, dim=1, keepdim=True)
        rprob = (logprob - logprob_max).exp() # relative prob
        logprob = torch.log(torch.mean(rprob, dim=1, keepdim=True) + 1e-10) + logprob_max # bsz x 1

        # return
        return logprob.mean()
