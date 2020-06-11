import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_nonlinear_func, expand_tensor, sample_laplace_noise, sample_unit_laplace_noise
from models.layers import MLP, WNMLP, Identity


def add_gaussian_noise(input, std):
    eps = torch.randn_like(input)
    return input + std*eps, eps

def add_uniform_noise(input, val):
    #raise NotImplementedError
    #eps = 2.*val*torch.rand_like(input) - val
    eps = torch.rand_like(input)
    return input + 2.*val*eps-val, eps

def add_laplace_noise(input, scale):
    eps = sample_unit_laplace_noise(shape=input.size(), dtype=input.dtype, device=input.device)
    return input + scale*eps, eps


class DAE(nn.Module):
    def __init__(self,
                 input_dim=2,
                 h_dim=1000,
                 std=0.1,
                 num_hidden_layers=1,
                 nonlinearity='tanh',
                 noise_type='gaussian',
                 #init=True,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.std = std
        self.num_hidden_layers = num_hidden_layers
        self.nonlinearity = nonlinearity
        self.noise_type = noise_type

        self.main = MLP(input_dim, h_dim, input_dim, use_nonlinearity_output=False, num_hidden_layers=num_hidden_layers, nonlinearity=nonlinearity)

    def add_noise(self, input, std=None):
        std = self.std if std is None else std
        if self.noise_type == 'gaussian':
            return add_gaussian_noise(input, std)
        elif self.noise_type == 'uniform':
            return add_uniform_noise(input, std)
        elif self.noise_type == 'laplace':
            return add_laplace_noise(input, std)
        else:
            raise NotImplementedError

    def loss(self, input, target):
        # recon loss (likelihood)
        recon_loss = F.mse_loss(input, target)#, reduction='sum')
        return recon_loss

    def forward(self, input, std=None):
        # init
        std = self.std if std is None else std
        batch_size = input.size(0)
        input = input.view(-1, self.input_dim)

        # add noise
        x_bar, eps = self.add_noise(input, std)

        # predict
        glogprob = self.main(x_bar)

        ''' get loss '''
        #loss = (std**2)*self.loss(std*glogprob, -eps)
        loss = self.loss(std*glogprob, -eps)

        # return
        return None, loss

    def glogprob(self, input, std=None):
        std = self.std if std is None else std
        batch_size = input.size(0)
        input = input.view(-1, self.input_dim)

        # predict
        glogprob = self.main(input)

        return glogprob

class ARDAE(nn.Module):
    def __init__(self,
                 input_dim=2,
                 h_dim=1000,
                 std=0.1,
                 num_hidden_layers=1,
                 nonlinearity='tanh',
                 noise_type='gaussian',
                 #init=True,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.std = std
        self.num_hidden_layers = num_hidden_layers
        self.nonlinearity = nonlinearity
        self.noise_type = noise_type
        #self.init = init

        self.main = MLP(input_dim+1, h_dim, input_dim, use_nonlinearity_output=False, num_hidden_layers=num_hidden_layers, nonlinearity=nonlinearity)

    def add_noise(self, input, std=None):
        std = self.std if std is None else std
        if self.noise_type == 'gaussian':
            return add_gaussian_noise(input, std)
        elif self.noise_type == 'uniform':
            return add_uniform_noise(input, std)
        elif self.noise_type == 'laplace':
            return add_laplace_noise(input, std)
        else:
            raise NotImplementedError

    def loss(self, input, target):
        # recon loss (likelihood)
        recon_loss = F.mse_loss(input, target)#, reduction='sum')
        return recon_loss

    def forward(self, input, std=None):
        # init
        batch_size = input.size(0)
        input = input.view(-1, self.input_dim)
        if std is None:
            std = input.new_zeros(batch_size, 1)
        else:
            assert torch.is_tensor(std)

        # add noise
        x_bar, eps = self.add_noise(input, std)

        # concat
        h = torch.cat([x_bar, std], dim=1)

        # predict
        glogprob = self.main(h)

        ''' get loss '''
        loss = self.loss(std*glogprob, -eps)

        # return
        return None, loss

    def glogprob(self, input, std=None):
        batch_size = input.size(0)
        input = input.view(-1, self.input_dim)
        if std is None:
            std = input.new_zeros(batch_size, 1)
        else:
            assert torch.is_tensor(std)

        # concat
        h = torch.cat([input, std], dim=1)

        # predict
        glogprob = self.main(h)

        return glogprob


class ConditionalDAE(nn.Module):
    def __init__(self,
                 input_dim=2, #10,
                 h_dim=128,
                 context_dim=2,
                 std=0.01,
                 num_hidden_layers=1,
                 nonlinearity='tanh',
                 noise_type='gaussian',
                 enc_input=True,
                 enc_ctx=True,
                 #init=True,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.context_dim = context_dim
        self.std = std
        self.num_hidden_layers = num_hidden_layers
        self.nonlinearity = nonlinearity
        self.noise_type = noise_type
        self.enc_input = enc_input
        if self.enc_input:
            inp_dim = h_dim
        else:
            inp_dim = input_dim
        self.enc_ctx = enc_ctx
        if self.enc_ctx:
            ctx_dim = h_dim
        else:
            ctx_dim = context_dim
        #self.init = init

        self.ctx_encode = Identity() if not self.enc_ctx \
                else MLP(context_dim, h_dim, h_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers-1, use_nonlinearity_output=True)
        self.inp_encode = Identity() if not self.enc_input \
                else MLP(input_dim,   h_dim, h_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers-1, use_nonlinearity_output=True)
        self.dae = MLP(inp_dim+ctx_dim, h_dim, input_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers, use_nonlinearity_output=False)

    def reset_parameters(self):
        nn.init.normal_(self.dae.fc.weight)

    def add_noise(self, input, std=None):
        std = self.std if std is None else std
        if self.noise_type == 'gaussian':
            return add_gaussian_noise(input, std)
        elif self.noise_type == 'uniform':
            return add_uniform_noise(input, std)
        elif self.noise_type == 'laplace':
            return add_laplace_noise(input, std)
        else:
            raise NotImplementedError

    def loss(self, input, target):
        # recon loss (likelihood)
        recon_loss = F.mse_loss(input, target)#, reduction='sum')
        return recon_loss

    def forward(self, input, context, std=None):
        # init
        assert input.dim() == 3 # bsz x ssz x x_dim
        assert context.dim() == 3 # bsz x 1 x ctx_dim
        std = self.std if std is None else std
        batch_size = input.size(0)
        sample_size = input.size(1)

        # reschape
        input = input.view(batch_size*sample_size, self.input_dim) # bsz*ssz x xdim
        _, context = expand_tensor(context, sample_size=sample_size, do_unsqueeze=False) # bsz*ssz x xdim
        #context = context.view(batch_size*sample_size, -1) # bsz*ssz x xdim

        # add noise
        x_bar, eps = self.add_noise(input, std)

        # encode
        ctx = self.ctx_encode(context)
        inp = self.inp_encode(x_bar)

        # concat
        h = torch.cat([inp, ctx], dim=1)

        # de-noise with context
        glogprob = self.dae(h)

        ''' get loss '''
        #loss = (std**2)*self.loss(std*glogprob, -eps)
        loss = self.loss(std*glogprob, -eps)

        # return
        return None, loss

    def glogprob(self, input, context, std=None):
        # init
        assert input.dim() == 3 # bsz x ssz x x_dim
        assert context.dim() == 3 # bsz x 1 x ctx_dim
        std = self.std if std is None else std
        batch_size = input.size(0)
        sample_size = input.size(1)

        # reschape
        input = input.view(batch_size*sample_size, self.input_dim) # bsz*ssz x xdim
        _, context = expand_tensor(context, sample_size=sample_size, do_unsqueeze=False) # bsz*ssz x xdim
        #context = context.view(batch_size*sample_size, -1) # bsz*ssz x xdim

        # encode
        ctx = self.ctx_encode(context)
        inp = self.inp_encode(input)

        # concat
        h = torch.cat([inp, ctx], dim=1)

        # de-noise with context
        glogprob = self.dae(h)

        return glogprob.view(batch_size, sample_size, self.input_dim)

class ConditionalARDAE(nn.Module):
    def __init__(self,
                 input_dim=2, #10,
                 h_dim=128,
                 context_dim=2,
                 std=0.01,
                 num_hidden_layers=1,
                 nonlinearity='tanh',
                 noise_type='gaussian',
                 enc_input=True,
                 enc_ctx=True,
                 #init=True,
                 std_method='default',
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.context_dim = context_dim
        self.std = std
        self.num_hidden_layers = num_hidden_layers
        self.nonlinearity = nonlinearity
        self.noise_type = noise_type
        self.enc_input = enc_input
        if self.enc_input:
            inp_dim = h_dim
        else:
            inp_dim = input_dim
        self.enc_ctx = enc_ctx
        if self.enc_ctx:
            ctx_dim = h_dim
        else:
            ctx_dim = context_dim

        self.ctx_encode = Identity() if not self.enc_ctx \
                else MLP(context_dim, h_dim, h_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers-1, use_nonlinearity_output=True)
        self.inp_encode = Identity() if not self.enc_input \
                else MLP(input_dim,   h_dim, h_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers-1, use_nonlinearity_output=True)
        self.dae = MLP(inp_dim+ctx_dim+1, h_dim, input_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers, use_nonlinearity_output=False)

    def reset_parameters(self):
        nn.init.normal_(self.dae.fc.weight)

    def add_noise(self, input, std=None):
        std = self.std if std is None else std
        if self.noise_type == 'gaussian':
            return add_gaussian_noise(input, std)
        elif self.noise_type == 'uniform':
            return add_uniform_noise(input, std)
        elif self.noise_type == 'laplace':
            return add_laplace_noise(input, std)
        else:
            raise NotImplementedError

    def loss(self, input, target):
        # recon loss (likelihood)
        recon_loss = F.mse_loss(input, target)#, reduction='sum')
        return recon_loss

    def forward(self, input, context, std=None, scale=None):
        # init
        assert input.dim() == 3 # bsz x ssz x x_dim
        assert context.dim() == 3 # bsz x 1 x ctx_dim
        batch_size = input.size(0)
        sample_size = input.size(1)
        if std is None:
            std = input.new_zeros(batch_size, sample_size, 1)
        else:
            assert torch.is_tensor(std)
        if scale is None:
            scale = 1.

        # reschape
        input = input.view(batch_size*sample_size, self.input_dim) # bsz*ssz x xdim
        _, context = expand_tensor(context, sample_size=sample_size, do_unsqueeze=False) # bsz*ssz x xdim
        #context = context.view(batch_size*sample_size, -1) # bsz*ssz x xdim
        std = std.view(batch_size*sample_size, 1)

        # add noise
        x_bar, eps = self.add_noise(input, std)

        # encode
        ctx = self.ctx_encode(context)
        inp = self.inp_encode(x_bar)

        # concat
        h = torch.cat([inp, ctx, std], dim=1)

        # de-noise with context
        glogprob = self.dae(h)

        ''' get loss '''
        #loss = (std**2)*self.loss(std*glogprob, -eps)
        loss = self.loss(std*glogprob, -eps)

        # return
        return None, loss

    def glogprob(self, input, context, std=None, scale=None):
        # init
        assert input.dim() == 3 # bsz x ssz x x_dim
        assert context.dim() == 3 # bsz x 1 x ctx_dim
        #std = self.std if std is None else std
        batch_size = input.size(0)
        sample_size = input.size(1)
        if std is None:
            std = input.new_zeros(batch_size*sample_size, 1)
        else:
            assert torch.is_tensor(std)
        if scale is None:
            scale = 1.

        # reschape
        input = input.view(batch_size*sample_size, self.input_dim) # bsz*ssz x xdim
        _, context = expand_tensor(context, sample_size=sample_size, do_unsqueeze=False) # bsz*ssz x xdim
        #context = context.view(batch_size*sample_size, -1) # bsz*ssz x xdim
        std = std.view(batch_size*sample_size, 1)

        # encode
        ctx = self.ctx_encode(context)
        inp = self.inp_encode(input)

        # concat
        h = torch.cat([inp, ctx, std], dim=1)

        # de-noise with context
        glogprob = self.dae(h)

        return glogprob.view(batch_size, sample_size, self.input_dim)
