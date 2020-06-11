import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.distributions import MultivariateNormal

from models.layers import Identity, MLP
from models.reparam import NormalDistributionLinear
from utils import loss_kld_gaussian, loss_kld_gaussian_vs_gaussian, loss_recon_gaussian, loss_recon_bernoulli_with_logit, normal_energy_func
from utils import logprob_gaussian, get_covmat
from utils import get_nonlinear_func
from utils import conv_out_size, deconv_out_size

from models.vae.conv import Decoder
from models.vae.auxconv import AuxEncoder, AuxDecoder
from models.vae.auxconv import Encoder as _Encoder
from models.vae.auxconv import weight_init

from utils import expand_tensor
from utils import cond_jac_clamping_loss


def sample_noise(sz, device):
    eps = torch.randn(*sz).to(device)
    return eps

def sample_gaussian(mu, logvar, _std=None, eps=None):
    if _std is None:
        _std = 1.
    std = _std*torch.exp(0.5*logvar)

    if eps is None:
        eps = torch.randn_like(std)
    return mu + std * eps, eps

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
                 input_height=28,
                 input_channels=1,
                 z0_dim=100,
                 z_dim=32,
                 nonlinearity='softplus',
                 #clip_z0_logvar=None,
                 #clip_z_logvar=None,
                 ):
        super().__init__()
        self.input_height = input_height
        self.input_channels = input_channels
        self.z0_dim = z0_dim
        self.z_dim = z_dim
        self.nonlinearity = nonlinearity
        #self.clip_z0_logvar = clip_z0_logvar
        #self.clip_z_logvar = clip_z_logvar

        self.aux_encode = AuxEncoder(input_height, input_channels, z0_dim, nonlinearity=nonlinearity)#, clip_logvar=clip_z0_logvar)
        self.encode = _Encoder(input_height, input_channels, z0_dim, z_dim, nonlinearity=nonlinearity)#, clip_logvar=clip_z_logvar)

    def _forward_w_eps(self, input, std=None, nz=1, eps0=None, eps=None):
        batch_size = input.size(0)
        input = input.view(batch_size, self.input_channels, self.input_height, self.input_height)

        # gen eps0, eps
        if eps0 is None:
            eps0 = sample_noise(sz=[batch_size*nz, self.z0_dim], device=input.device)
        if eps is None:
            eps = sample_noise(sz=[batch_size*nz, 1, self.z_dim], device=input.device)

        # aux encode
        _, mu_qz0, logvar_qz0, h0 = self.aux_encode(input, _std=std)
        mu_qz0 = convert_2d_3d_tensor(mu_qz0, sample_size=nz).view(batch_size*nz, -1)
        logvar_qz0 = convert_2d_3d_tensor(logvar_qz0, sample_size=nz).view(batch_size*nz, -1)
        z0, _ = sample_gaussian(mu_qz0, logvar_qz0, _std=std, eps=eps0)#.view(batch_size*nz, -1)

        # encode
        _, mu_qz, logvar_qz, h = self.encode(input, z0, nz=nz)

        # expand
        mu_qz = convert_2d_3d_tensor(mu_qz, sample_size=1)
        logvar_qz = convert_2d_3d_tensor(logvar_qz, sample_size=1)

        # sample z
        z, _ = sample_gaussian(mu_qz, logvar_qz, _std=std, eps=eps)
        z = z.view(batch_size, nz, -1)

        return (z0, mu_qz0, logvar_qz0, eps0,
                z, mu_qz, logvar_qz, eps,
                (h0, h),
                )

    def _forward(self, input, std=None, nz=1):
        batch_size = input.size(0)
        input = input.view(batch_size, self.input_channels, self.input_height, self.input_height)

        # gen eps0, eps
        eps0 = sample_noise(sz=[batch_size*nz, self.z0_dim], device=input.device)
        eps = sample_noise(sz=[batch_size*nz, 1, self.z_dim], device=input.device)

        return self._forward_w_eps(input, std=std, nz=nz, eps0=eps0, eps=eps)

    def forward(self, x, std=None, nz=1):
        (z0, mu_qz0, logvar_qz0, eps0,
         z, mu_qz, logvar_qz, eps,
         _,
         ) = self._forward(x, std=std, nz=nz)
        return z # bsz x nz x z_dim

    def forward_hidden(self, x, std=None, nz=1):
        assert nz == 1
        (z0, mu_qz0, logvar_qz0, eps0,
         z, mu_qz, logvar_qz, eps,
         h0_h,
         ) = self._forward(x, std=std, nz=nz)
        h = torch.cat(h0_h, dim=1)
        return h # bsz*nz x h_dim

class ImplicitPosteriorVAE(nn.Module):
    def __init__(self,
                 energy_func=normal_energy_func,
                 input_height=28,
                 input_channels=1,
                 z0_dim=100,
                 h_dim=300,
                 z_dim=32,
                 nonlinearity='softplus',
                 #clip_z0_logvar=None,
                 #clip_z_logvar=None,
                 do_xavier=True,
                 #do_m5bias=False,
                 ):
        super().__init__()
        self.energy_func = energy_func
        self.input_height = input_height
        self.input_channels = input_channels
        self.z0_dim = z0_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.latent_dim = z_dim # for ais
        self.nonlinearity = nonlinearity
        #self.num_hidden_layers = num_hidden_layers
        #clip_z0_logvar = None if clip_z0_logvar == 'none' else clip_z0_logvar
        #clip_z_logvar = None if clip_z_logvar == 'none' else clip_z_logvar
        #self.clip_z0_logvar = clip_z0_logvar
        #self.clip_z_logvar = clip_z_logvar
        #self.enc_type = enc_type
        #assert enc_type in ['simple']
        self.do_xavier = do_xavier
        #self.do_m5bias = do_m5bias

        self.encode = Encoder(input_height, input_channels, z0_dim, z_dim, nonlinearity=nonlinearity)#, clip_z0_logvar=clip_z0_logvar, clip_z_logvar=clip_z_logvar)
        self.decode = Decoder(input_height, input_channels, z_dim, nonlinearity=nonlinearity)
        #self.aux_decode = AuxDecoder(input_height, input_channels, z_dim, z0_dim, nonlinearity=nonlinearity)
        self.reset_parameters()

    def reset_parameters(self):
        if self.do_xavier:
            self.apply(weight_init)
        #if self.do_m5bias:
        #    torch.nn.init.constant_(self.decode.reparam.logit_fn.bias, -5)

    def primary_loss(self, z, logit_px, target_x):
        # loss from energy func
        prior_loss = self.energy_func(z.view(-1, self.z_dim))

        # recon loss (neg likelihood): -log p(x|z)
        recon_loss = loss_recon_bernoulli_with_logit(logit_px, target_x, do_sum=False)

        return recon_loss, prior_loss

    def aux_loss(self, mu_pz0, logvar_pz0, target_z0):
        # aux dec loss: -log r(z0|z,x)
        aux_recon_loss = loss_recon_gaussian(mu_pz0, logvar_pz0, target_z0.view(-1, self.z0_dim), do_sum=False)
        return aux_recon_loss

    def jac_clamping_loss(self, input, z, eps0, eps, std, nz, eta_min, p=2, EPS=1.):
        raise NotImplementedError

    def loss(self,
             z,
             #mu_qz0, logvar_qz0,
             #mu_pz0, logvar_pz0, target_z0,
             logit_px, target_x,
             beta=1.0,
             ):

        # primary loss to train model
        recon_loss, prior_loss = self.primary_loss(z, logit_px, target_x)

        # additional training apart from the model training for evaluation
        #aux_kld_loss = self.aux_loss(mu_qz0, logvar_qz0, mu_pz0, logvar_pz0)
        #aux_recon_loss = self.aux_loss(mu_pz0, logvar_pz0, target_z0)

        # add loss
        loss = recon_loss + beta*prior_loss# + aux_recon_loss
        return loss.mean(), recon_loss.mean(), prior_loss.mean()#, aux_recon_loss.mean()

    def forward_hidden(self, input, std=None, nz=1):
        # init
        #_nz = int(math.sqrt(nz))
        batch_size = input.size(0)
        input = input.view(batch_size, self.input_channels, self.input_height, self.input_height)

        # encode
        (z0, mu_qz0, logvar_qz0, eps0,
         z, mu_qz, logvar_qz, eps,
         _,
         ) = self.encode._forward(input, std=std, nz=nz)

        return z

    def forward(self, input, beta=1.0, eta=0.0, lmbd=0.0, std=None, nz=1):
        # init
        #_nz = int(math.sqrt(nz))
        batch_size = input.size(0)
        input = input.view(batch_size, self.input_channels, self.input_height, self.input_height)
        input_expanded = convert_4d_5d_tensor(input, sample_size=nz)
        input_expanded_flattened = input_expanded.view(batch_size*nz, self.input_channels, self.input_height, self.input_height)

        ## aux encode
        #z0, mu_qz0, logvar_qz0 = self.encode.aux_encode(input)

        ## encode
        #z, mu_qz, logvar_qz = self.encode.encode(input, z0)
        (z0, mu_qz0, logvar_qz0, eps0,
         z, mu_qz, logvar_qz, eps,
         _,
         ) = self.encode._forward(input, std=std, nz=nz)

        # z flattten
        z0_expanded_flattened = convert_2d_3d_tensor(z0, sample_size=1).view(batch_size*nz, -1)
        z_flattened = z.view(batch_size*nz, -1)

        ## aux decode
        #(_, mu_pz0, logvar_pz0
        # ) = self.aux_decode(
        #         input_expanded_flattened,
        #         z_flattened.detach(),
        #         )

        # decode
        x, logit_px = self.decode(z_flattened)

        ''' get loss '''
        if lmbd > 0:
            raise NotImplementedError
            jaclmp_loss = lmbd*self.jac_clamping_loss(input, z, eps0, eps, std=std, nz=nz, eta_min=eta)
        else:
            jaclmp_loss = 0
        #loss, recon_loss, prior_loss, aux_recon_loss = self.loss(
        loss, recon_loss, prior_loss = self.loss(
                #mu_z, logvar_z,
                z_flattened,
                #mu_pz0, logvar_pz0, z0_expanded_flattened.detach(),
                logit_px, input_expanded_flattened,
                beta=beta,
                )
        loss += jaclmp_loss

        # return
        return x, torch.sigmoid(logit_px), z, loss, recon_loss.detach(), prior_loss.detach() #,aux_recon_loss.detach()

    def generate(self, batch_size=1):
        # init mu_z and logvar_z (as unit normal dist)
        weight = next(self.parameters())
        mu_z = weight.new_zeros(batch_size, self.z_dim)
        logvar_z = weight.new_zeros(batch_size, self.z_dim)

        # sample z (from unit normal dist)
        z, _ = sample_gaussian(mu_z, logvar_z) # sample z

        # decode
        output, logit_px = self.decode(z)

        # return
        return output, torch.sigmoid(logit_px), z

    def logprob(self, input, sample_size=128, z=None, std=None):
        return self.logprob_w_cov_gaussian_posterior(input, sample_size, z, std)

    def logprob_w_cov_gaussian_posterior(self, input, sample_size=128, z=None, std=None):
        # init
        batch_size = input.size(0)
        input = input.view(batch_size, self.input_channels, self.input_height, self.input_height)
        assert sample_size >= 2*self.z_dim
        #assert int(math.sqrt(sample_size))**2 == sample_size

        ''' get z and pseudo log q(newz|x) '''
        #z, newz = [], []
        #logposterior = []
        #inp = self.encode._forward_inp(input).detach()
        #for i in range(batch_size):
        #    _inp = inp[i:i+1, :].expand(sample_size, inp.size(1))
        #    _nos = self.encode._forward_nos(sample_size, std=std, device=input.device).detach()
        #    _z = self.encode._forward_all(_inp, _nos) # ssz x zdim
        #    z += [_z.detach().unsqueeze(0)]
        #z = torch.cat(z, dim=0) # bsz x ssz x zdim
        #_nz = int(math.sqrt(sample_size))
        _, _, _, _, z, _, _, _, _ = self.encode._forward(input, std=std, nz=sample_size) # bsz x ssz x zdim
        newz = []
        logposterior = []
        eye = torch.eye(self.z_dim, device=z.device)
        mu_qz = torch.mean(z, dim=1) # bsz x zdim
        for i in range(batch_size):
            _cov_qz = get_covmat(z[i, :, :]) + 1e-5*eye
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

    #def logprob(self, input, sample_size=128, z0_sample_size=1, z=None):
    #    # init
    #    batch_size = input.size(0)
    #    sample_size1 = z0_sample_size
    #    sample_size2 = sample_size
    #    input = input.view(batch_size, self.input_dim)

    #    ''' get - (log q(z|z0,x) + log q(z0|z) - log p(z0|z,x) - log p(z)) '''
    #    ''' get log q(z0|x) '''
    #    _, mu_qz0, logvar_qz0 = self.encode.aux_encode(input)
    #    mu_qz0 = mu_qz0.unsqueeze(1).expand(batch_size, sample_size1, self.z0_dim).contiguous().view(batch_size*sample_size1, self.z0_dim) # bsz*ssz1 x z0_dim
    #    logvar_qz0 = logvar_qz0.unsqueeze(1).expand(batch_size, sample_size1, self.z0_dim).contiguous().view(batch_size*sample_size1, self.z0_dim) # bsz*ssz1 x z0_dim
    #    z0 = self.encode.aux_encode.sample(mu_qz0, logvar_qz0) # bsz*ssz1 x z0_dim
    #    log_qz0 = logprob_gaussian(mu_qz0, logvar_qz0, z0, do_unsqueeze=False, do_mean=False)
    #    log_qz0 = torch.sum(log_qz0.view(batch_size, sample_size1, self.z0_dim), dim=2) # bsz x ssz1
    #    log_qz0 = log_qz0.unsqueeze(2).expand(batch_size, sample_size1, sample_size2).contiguous().view(batch_size, sample_size1*sample_size2) # bsz x ssz1*ssz2

    #    ''' get log q(z|z0,x) '''
    #    # forward
    #    nos = self.encode.encode._forward_nos(z0) # bsz*ssz1 x z0_dim
    #    inp = self.encode.encode._forward_inp(input) # bsz x z0_dim
    #    inp = inp.unsqueeze(1).expand(batch_size, sample_size1, -1).contiguous().view(batch_size*sample_size1, -1)
    #    _, mu_qz, logvar_qz = self.encode.encode._forward_all(inp, nos) # bsz*ssz1 x z_dim
    #    mu_qz = mu_qz.detach().repeat(1, sample_size2).view(batch_size*sample_size1, sample_size2, self.z_dim)
    #    logvar_qz = logvar_qz.detach().repeat(1, sample_size2).view(batch_size*sample_size1, sample_size2, self.z_dim)
    #    z = self.encode.encode.sample(mu_qz, logvar_qz) # bsz x ssz1 x ssz2 x z_dim
    #    log_qz = logprob_gaussian(mu_qz, logvar_qz, z, do_unsqueeze=False, do_mean=False)
    #    log_qz = torch.sum(log_qz.view(batch_size, sample_size1*sample_size2, self.z_dim), dim=2) # bsz x ssz1*ssz2

    #    ''' get log p(z0|z,x) '''
    #    # encode
    #    _z0 = z0.unsqueeze(1).expand(batch_size*sample_size1, sample_size2, self.z0_dim).contiguous().view(batch_size, sample_size1, sample_size2, self.z0_dim)
    #    ltt = self.aux_decode._forward_ltt(z.view(-1, self.z_dim)) # bsz*ssz1*ssz2 x z_dim
    #    inp = self.aux_decode._forward_inp(input) # bsz x inp_dim
    #    inp = inp.unsqueeze(1).unsqueeze(1).expand(batch_size, sample_size1, sample_size2, -1).contiguous().view(batch_size*sample_size1*sample_size2, -1) # bsz*ss1*ssz2 x inp_dim
    #    _, mu_pz0, logvar_pz0 = self.aux_decode._forward_all(inp, ltt) # bsz*ssz1 x z_dim
    #    mu_pz0 = mu_pz0.view(batch_size, sample_size1, sample_size2, self.z0_dim)
    #    logvar_pz0 = logvar_pz0.view(batch_size, sample_size1, sample_size2, self.z0_dim)
    #    log_pz0 = logprob_gaussian(mu_pz0, logvar_pz0, _z0, do_unsqueeze=False, do_mean=False) # bsz x ssz1 x ssz2 xz0_dim
    #    log_pz0 = torch.sum(log_pz0.view(batch_size, sample_size1*sample_size2, self.z0_dim), dim=2) # bsz x ssz1*ssz2

    #    ''' get log p(z) '''
    #    # get prior (as unit normal dist)
    #    mu_pz = input.new_zeros(batch_size*sample_size1, sample_size2, self.z_dim)
    #    logvar_pz = input.new_zeros(batch_size*sample_size1, sample_size2, self.z_dim)
    #    log_pz = logprob_gaussian(mu_pz, logvar_pz, z, do_unsqueeze=False, do_mean=False)
    #    log_pz = torch.sum(log_pz.view(batch_size, sample_size1*sample_size2, self.z_dim), dim=2) # bsz x ssz1*ssz2

    #    ''' get log p(x|z) '''
    #    # decode
    #    _input = input.unsqueeze(1).unsqueeze(1).expand(
    #            batch_size, sample_size1, sample_size2, self.input_dim) # bsz x ssz1 x ssz2 x input_dim
    #    _z = z.view(-1, self.z_dim)
    #    #_, mu_x, logvar_x = self.decode(_z) # bsz*ssz1*ssz2 x zdim
    #    #mu_x = mu_x.view(batch_size, sample_size1, sample_size2, self.input_dim)
    #    #logvar_x = logvar_x.view(batch_size, sample_size1, sample_size2, self.input_dim)
    #    #loglikelihood = logprob_gaussian(mu_x, logvar_x, _input, do_unsqueeze=False, do_mean=False)
    #    _, logit_px = self.decode(_z) # bsz*ssz1*ssz2 x zdim
    #    logit_px = logit_px.view(batch_size, sample_size1, sample_size2, self.input_dim)
    #    loglikelihood = -F.binary_cross_entropy_with_logits(logit_px, _input, reduction='none')
    #    loglikelihood = torch.sum(loglikelihood.view(batch_size, sample_size1*sample_size2, self.input_dim), dim=2) # bsz x ssz1*ssz2

    #    ''' get log p(x|z)p(z)/q(z|x) '''
    #    logprob = loglikelihood + log_pz + log_pz0 - log_qz - log_qz0 # bsz x ssz1*ssz2
    #    logprob_max, _ = torch.max(logprob, dim=1, keepdim=True)
    #    rprob = (logprob - logprob_max).exp() # relative prob
    #    logprob = torch.log(torch.mean(rprob, dim=1, keepdim=True) + 1e-10) + logprob_max # bsz x 1

    #    # return
    #    return logprob.mean()
