import math
import numpy as np
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.distributions import MultivariateNormal

from models.layers import Identity, MLP, WNMLP, ContextConcatMLP, ContextScaleMLP, ContextWNScaleMLP, ContextSPScaleMLP, ContextSPWNScaleMLP, ContextBilinearMLP, ContextWNBilinearMLP, ContextSWNBilinearMLP, ContextResMLP
from models.reparam import BernoulliDistributionLinear
from utils import loss_recon_bernoulli_with_logit, normal_energy_func
from utils import logprob_gaussian, get_covmat
from utils import get_nonlinear_func
from utils import expand_tensor
from utils import cond_jac_clamping_loss


def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        #torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def sample_gaussian(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + std * eps

def convert_2d_3d_tensor(input, sample_size):
    assert input.dim() == 2
    input_expanded, _ = expand_tensor(input, sample_size, do_unsqueeze=True)
    return input_expanded


class Encoder(nn.Module):
    def __init__(self,
                 input_dim=2,
                 noise_dim=2,
                 h_dim=64,
                 z_dim=2,
                 nonlinearity='softplus',
                 num_hidden_layers=1,
                 std=1.,
                 init='none', #'gaussian',
                 enc_noise=False, #True,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.noise_dim = noise_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.nonlinearity = nonlinearity
        self.num_hidden_layers = num_hidden_layers
        self.std = std
        self.init = init
        self.enc_noise = enc_noise
        #ctx_dim = noise_dim if not enc_noise else h_dim

        #self.inp_encode = MLP(input_dim=input_dim, hidden_dim=h_dim, output_dim=h_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers-1, use_nonlinearity_output=True)
        #self.nos_encode = Identity() if not enc_noise \
        #        else MLP(input_dim=noise_dim, hidden_dim=h_dim, output_dim=h_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers-1, use_nonlinearity_output=True)
        #self.fc  = ContextConcatMLP(input_dim=h_dim, context_dim=ctx_dim, hidden_dim=h_dim, output_dim=z_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers, use_nonlinearity_output=False)

    def reset_parameters(self):
        raise NotImplementedError

    def sample_noise(self, batch_size, std=None, device=None):
        std = std if std is not None else self.std
        device = device if device is not None else next(self.parameters).device
        eps = torch.randn(batch_size, self.noise_dim).to(device)
        return std * eps

    def _forward_inp(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.input_dim)

        # rescale
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
        raise NotImplementedError
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

class ConcatEncoder(Encoder):
    def __init__(self,
                 input_dim=2,
                 noise_dim=2,
                 h_dim=64,
                 z_dim=2,
                 nonlinearity='softplus',
                 num_hidden_layers=1,
                 std=1.,
                 init='none', #'gaussian',
                 enc_noise=False,
                 ):
        super().__init__(
        input_dim = input_dim,
        noise_dim = noise_dim,
        h_dim = h_dim,
        z_dim = z_dim,
        nonlinearity = nonlinearity,
        num_hidden_layers = num_hidden_layers,
        std = std,
        init = init,
        enc_noise = enc_noise,
        )
        nos_dim = noise_dim if not enc_noise else h_dim

        self.inp_encode = MLP(input_dim=input_dim, hidden_dim=h_dim, output_dim=h_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers, use_nonlinearity_output=True)
        self.nos_encode = Identity() if not enc_noise \
                else MLP(input_dim=noise_dim, hidden_dim=h_dim, output_dim=h_dim, nonlinearity=nonlinearity, num_hidden_layers=0, use_nonlinearity_output=True)
        self.fc  = MLP(input_dim=h_dim+nos_dim, hidden_dim=h_dim, output_dim=z_dim, nonlinearity=nonlinearity, num_hidden_layers=1, use_nonlinearity_output=False)

        if self.init == 'gaussian':
            self.reset_parameters()
        else:
            pass

    def reset_parameters(self):
        nn.init.normal_(self.fc.fc.weight)

    def _forward_all(self, inp, nos):
        #z = self.fc(inp, nos)
        inp_nos = torch.cat([inp, nos], dim=1)
        z = self.fc(inp_nos)
        return z

class Decoder(nn.Module):
    def __init__(self,
                 input_dim=784,
                 h_dim=300,
                 z_dim=32,
                 nonlinearity='softplus',
                 num_hidden_layers=1,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.nonlinearity = nonlinearity
        self.num_hidden_layers = num_hidden_layers

        self.main = MLP(input_dim=z_dim, hidden_dim=h_dim, output_dim=h_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers, use_nonlinearity_output=True)
        self.reparam = BernoulliDistributionLinear(h_dim, input_dim)

    def sample(self, logit):
        return self.reparam.sample_logistic_sigmoid(logit)

    def forward(self, z):
        batch_size = z.size(0)
        z = z.view(batch_size, -1)

        # forward
        h = self.main(z)
        logit = self.reparam(h)

        # sample
        x = self.sample(logit)

        return x, logit

class ImplicitPosteriorVAE(nn.Module):
    def __init__(self,
                 energy_func=normal_energy_func,
                 input_dim=784,
                 noise_dim=100,
                 h_dim=300,
                 z_dim=32,
                 nonlinearity='softplus',
                 num_hidden_layers=1,
                 init='gaussian',
                 enc_type='concat',
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
        self.enc_type = enc_type
        assert enc_type in ['concat']

        if enc_type == 'concat':
            self.encode = ConcatEncoder(input_dim, noise_dim, h_dim, z_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers+1, init=init)
        else:
            raise NotImplementedError
        self.decode = Decoder(input_dim, h_dim, z_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers)
        self.reset_parameters()

    def reset_parameters(self):
        #self.apply(weight_init)
        self.decode.apply(weight_init)
        #torch.nn.init.constant_(self.decode.reparam.logit_fn.bias, -5)
        if self.init == 'gaussian':
            self.encode.reset_parameters()

    def loss(self, z, logit_x, target_x, beta=1.0):
        # loss from energy func
        prior_loss = self.energy_func(z.view(-1, self.z_dim))

        # recon loss (neg likelihood): -log p(x|z)
        recon_loss = loss_recon_bernoulli_with_logit(logit_x, target_x.view(-1, self.input_dim), do_sum=False)

        # add loss
        loss = recon_loss + beta*prior_loss
        return loss.mean(), recon_loss.mean(), prior_loss.mean()

    def jac_clamping_loss(self, input, z, eps, std, nz, eta_min, p=2, EPS=1.):
        raise NotImplementedError

    def forward_hidden(self, input, std=None, nz=1):
        # init
        batch_size = input.size(0)
        input = input.view(batch_size, self.input_dim)

        # gen noise source
        eps = self.encode.sample_noise(batch_size*nz, std=std, device=input.device)

        # sample z
        z = self.encode(input, noise=eps, std=std, nz=nz)

        return z

    def forward(self, input, beta=1.0, eta=0.0, lmbd=0.0, std=None, nz=1):
        # init
        batch_size = input.size(0)
        input = input.view(batch_size, self.input_dim)
        input_expanded = convert_2d_3d_tensor(input, sample_size=nz)
        input_expanded_flattened = input_expanded.view(batch_size*nz, -1)
        #target = input.unsqueeze(1).expand(-1, nz, -1).contiguous().view(batch_size*nz, -1)

        # gen noise source
        eps = self.encode.sample_noise(batch_size*nz, std=std, device=input.device)

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

    def logprob_w_kde_posterior(self, input, sample_size=128, z=None, std=None):
        # init
        batch_size = input.size(0)
        input = input.view(batch_size, self.input_dim)
        assert sample_size >= 2*self.z_dim

        ''' get z and pseudo log q(newz|x) '''
        z, newz = [], []
        logposterior = []
        inp = self.encode._forward_inp(input).detach()
        for i in range(batch_size):
            _inp = inp[i:i+1, :].expand(sample_size, inp.size(1))
            _nos = self.encode._forward_nos(sample_size, std=std, device=input.device).detach()
            _z = self.encode._forward_all(_inp, _nos) # ssz x zdim
            z += [_z.detach().unsqueeze(0)]
        z = torch.cat(z, dim=0) # bsz x ssz x zdim
        for i in range(batch_size):
            _z = z[i, :, :].cpu().numpy().T # zdim x ssz
            kernel = stats.gaussian_kde(_z)
            _newz = kernel.resample(sample_size) # zdim x ssz
            _logposterior = kernel.logpdf(_newz) # ssz

            _newz = torch.from_numpy(_newz.T).float().to(input.device) # ssz x zdim
            _logposterior = torch.from_numpy(_logposterior).float().to(input.device) # ssz
            newz += [_newz.unsqueeze(0)]
            logposterior += [_logposterior.unsqueeze(0)]
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
        _input = input.unsqueeze(1).expand(batch_size, sample_size, self.input_dim) # bsz x ssz x input_dim
        loglikelihood = -F.binary_cross_entropy_with_logits(logit_x, _input, reduction='none')
        loglikelihood = torch.sum(loglikelihood, dim=2) # bsz x ssz

        ''' get log p(x|z)p(z)/q(z|x) '''
        logprob = loglikelihood + logprior - logposterior # bsz x ssz
        logprob_max, _ = torch.max(logprob, dim=1, keepdim=True)
        rprob = (logprob - logprob_max).exp() # relative prob
        logprob = torch.log(torch.mean(rprob, dim=1, keepdim=True) + 1e-10) + logprob_max # bsz x 1

        # return
        return logprob.mean()

    def logprob_w_cov_gaussian_posterior(self, input, sample_size=128, z=None, std=None):
        # init
        batch_size = input.size(0)
        input = input.view(batch_size, self.input_dim)
        assert sample_size >= 2*self.z_dim

        ''' get z and pseudo log q(newz|x) '''
        z, newz = [], []
        #cov_qz, rv_z = [], []
        logposterior = []
        inp = self.encode._forward_inp(input).detach()
        #for i in range(sample_size):
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

            #cov_qz += [_cov_qz.unsqueeze(0)]
            #rv_z += [_rv_z]
            newz += [_newz]
            logposterior += [_logposterior]
        #cov_qz = torch.cat(cov_qz, dim=0) # bsz x zdim x zdim
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
        _input = input.unsqueeze(1).expand(batch_size, sample_size, self.input_dim) # bsz x ssz x input_dim
        loglikelihood = -F.binary_cross_entropy_with_logits(logit_x, _input, reduction='none')
        loglikelihood = torch.sum(loglikelihood, dim=2) # bsz x ssz

        ''' get log p(x|z)p(z)/q(z|x) '''
        logprob = loglikelihood + logprior - logposterior # bsz x ssz
        logprob_max, _ = torch.max(logprob, dim=1, keepdim=True)
        rprob = (logprob - logprob_max).exp() # relative prob
        logprob = torch.log(torch.mean(rprob, dim=1, keepdim=True) + 1e-10) + logprob_max # bsz x 1

        # return
        return logprob.mean()

    def logprob_w_diag_gaussian_posterior(self, input, sample_size=128, z=None, std=None):
        # init
        batch_size = input.size(0)
        input = input.view(batch_size, self.input_dim)

        ''' get z '''
        z = []
        for i in range(sample_size):
            _z = self.encode(input, std=std)
            _z_flattened = _z.view(_z.size(1)*_z.size(2), -1)
            z += [_z_flattened.detach().unsqueeze(1)]
        z = torch.cat(z, dim=1) # bsz x ssz x zdim
        mu_qz = torch.mean(z, dim=1)
        logvar_qz = torch.log(torch.var(z, dim=1) + 1e-10)

        ''' get pseudo log q(z|x) '''
        mu_qz = mu_qz.detach().repeat(1, sample_size).view(batch_size, sample_size, self.z_dim)
        logvar_qz = logvar_qz.detach().repeat(1, sample_size).view(batch_size, sample_size, self.z_dim)
        newz = sample_gaussian(mu_qz, logvar_qz)
        logposterior = logprob_gaussian(mu_qz, logvar_qz, newz, do_unsqueeze=False, do_mean=False)
        logposterior = torch.sum(logposterior.view(batch_size, sample_size, self.z_dim), dim=2) # bsz x ssz

        ''' get log p(z) '''
        # get prior (as unit normal dist)
        mu_pz = input.new_zeros(batch_size, sample_size, self.z_dim)
        logvar_pz = input.new_zeros(batch_size, sample_size, self.z_dim)
        logprior = logprob_gaussian(mu_pz, logvar_pz, newz, do_unsqueeze=False, do_mean=False)
        logprior = torch.sum(logprior.view(batch_size, sample_size, self.z_dim), dim=2) # bsz x ssz

        ''' get log p(x|z) '''
        # decode
        logit_x = []
        for i in range(sample_size):
            _, _logit_x = self.decode(newz[:, i, :])
            logit_x += [_logit_x.detach().unsqueeze(1)]
        logit_x = torch.cat(logit_x, dim=1) # bsz x ssz x input_dim
        _input = input.unsqueeze(1).expand(batch_size, sample_size, self.input_dim) # bsz x ssz x input_dim
        loglikelihood = -F.binary_cross_entropy_with_logits(logit_x, _input, reduction='none')
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
        logit_x = []
        for i in range(sample_size):
            _, _logit_x = self.decode(z[:, i, :])
            logit_x += [_logit_x.detach().unsqueeze(1)]
        logit_x = torch.cat(logit_x, dim=1) # bsz x ssz x input_dim
        _input = input.unsqueeze(1).expand(batch_size, sample_size, self.input_dim) # bsz x ssz x input_dim
        loglikelihood = -F.binary_cross_entropy_with_logits(logit_x, _input, reduction='none')
        loglikelihood = torch.sum(loglikelihood, dim=2) # bsz x ssz

        ''' get log p(x) '''
        logprob = loglikelihood # bsz x ssz
        logprob_max, _ = torch.max(logprob, dim=1, keepdim=True)
        rprob = (logprob-logprob_max).exp() # relative prob
        logprob = torch.log(torch.mean(rprob, dim=1, keepdim=True) + 1e-10) + logprob_max # bsz x 1

        # return
        return logprob.mean()
