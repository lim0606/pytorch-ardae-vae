import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_nonlinear_func, expand_tensor
from models.layers import Identity, MLP, WNMLP, ContextScaleMLP


def add_gaussian_noise(input, std):
    eps = torch.randn_like(input)
    return input + std*eps

def add_uniform_noise(input, val):
    eps = 2.*val*torch.rand_like(input) - val
    return input + eps


class DAE(nn.Module):
    def __init__(self,
                 input_dim=2,
                 h_dim=1000,
                 std=0.1,
                 num_hidden_layers=1,
                 nonlinearity='tanh',
                 noise_type='gaussian',
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
        x_bar = self.add_noise(input, std)

        # predict
        x_recon = self.main(x_bar)

        ''' get loss '''
        loss = self.loss(x_recon, input)

        # return
        return x_recon, loss

    def glogprob(self, input, std=None):
        # init
        std = self.std if std is None else std
        sz = input.size()
        batch_size = input.size(0)
        input = input.view(batch_size, self.input_dim)

        input_recon = self.main(input)
        var = std**2
        grad = (input_recon - input) / var
        return grad.view(sz)


class ConditionalDAE(nn.Module):
    def __init__(self,
                 input_dim=2, #10,
                 h_dim=128,
                 context_dim=2,
                 std=0.1,
                 num_hidden_layers=1,
                 nonlinearity='tanh',
                 noise_type='gaussian',
                 enc_input=False,
                 enc_ctx=True,
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
        if enc_ctx:
            ctx_dim = h_dim
        else:
            ctx_dim = context_dim

        self.ctx_encode = Identity() if not self.enc_ctx \
                else MLP(context_dim, h_dim, h_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers-1, use_nonlinearity_output=True)
        self.inp_encode = Identity() if not self.enc_input \
                else MLP(input_dim,   h_dim, h_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers-1, use_nonlinearity_output=True)
        self.dae = MLP(inp_dim+ctx_dim, h_dim, input_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers, use_nonlinearity_output=False)

    def add_noise(self, input, std=None):
        std = self.std if std is None else std
        if self.noise_type == 'gaussian':
            return add_gaussian_noise(input, std)
        elif self.noise_type == 'uniform':
            return add_uniform_noise(input, std)
        else:
            raise NotImplementedError

    def loss(self, input, target):
        # recon loss (likelihood)
        recon_loss = F.mse_loss(input, target)#, reduction='sum')
        return recon_loss

    def forward(self, input, context, std=None):
        # init
        assert input.dim() == 3 # bsz x ssz x x_dim
        assert context.dim() == 3 # bsz x ssz x ctx_dim
        std = self.std if std is None else std
        batch_size = input.size(0)
        sample_size = input.size(1)

        # reschape
        input = input.view(batch_size*sample_size, self.input_dim) # bsz*ssz x xdim
        context = context.view(batch_size*sample_size, -1) # bsz*ssz x xdim

        # add noise
        x_bar = self.add_noise(input, std)

        # encode
        ctx = self.ctx_encode(context)
        inp = self.inp_encode(x_bar)

        # concat
        h = torch.cat([inp, ctx], dim=1)

        # de-noise with context
        x_recon = self.dae(h)

        ''' get loss '''
        loss = self.loss(x_recon, input)

        # return
        return x_recon, loss

    def glogprob(self, input, context, std=None):
        # init
        assert input.dim() == 3 # bsz x ssz x x_dim
        assert context.dim() == 3 # bsz x ssz x ctx_dim
        std = self.std if std is None else std
        sz = input.size()
        batch_size = input.size(0)
        sample_size = input.size(1)

        # reschape
        input = input.view(batch_size*sample_size, self.input_dim) # bsz*ssz x xdim
        context = context.view(batch_size*sample_size, -1) # bsz*ssz x xdim

        # encode
        ctx = self.ctx_encode(context)
        inp = self.inp_encode(input)

        # concat
        h = torch.cat([inp, ctx], dim=1)

        # de-noise with context
        input_recon = self.dae(h)

        # grad
        var = std**2
        grad = (input_recon - input) / var # -loglikelihood
        return grad.view(sz)


#class WNDAE(nn.Module):
#    def __init__(self,
#                 input_dim=2,
#                 h_dim=1000,
#                 std=0.1,
#                 num_hidden_layers=1,
#                 nonlinearity='tanh',
#                 noise_type='gaussian',
#                 ):
#        super().__init__()
#        raise NotImplementedError
#        self.input_dim = input_dim
#        self.h_dim = h_dim
#        self.std = std
#        self.num_hidden_layers = num_hidden_layers
#        self.nonlinearity = nonlinearity
#        self.noise_type = noise_type
#
#        self.main = WNMLP(input_dim, h_dim, input_dim, use_nonlinearity_output=False, num_hidden_layers=num_hidden_layers, nonlinearity=nonlinearity)
#
#    def add_noise(self, input, std=None):
#        std = self.std if std is None else std
#        if self.noise_type == 'gaussian':
#            return add_gaussian_noise(input, std)
#        elif self.noise_type == 'uniform':
#            return add_uniform_noise(input, std)
#        else:
#            raise NotImplementedError
#
#    def loss(self, input, target):
#        # recon loss (likelihood)
#        recon_loss = F.mse_loss(input, target)#, reduction='sum')
#        return recon_loss
#
#    def forward(self, input, std=None):
#        # init
#        batch_size = input.size(0)
#        input = input.view(-1, self.input_dim)
#
#        # add noise
#        x_bar = self.add_noise(input, std)
#
#        ## encode
#        #z = self.encode(x_bar)
#
#        ## decode
#        #x_recon = self.decode(z)
#        x_recon = self.main(x_bar)
#
#        ''' get loss '''
#        loss = self.loss(x_recon, input)
#
#        # return
#        return x_recon, loss
#
#    def glogprob(self, input, std=None):
#        # init
#        sz = input.size()
#        batch_size = input.size(0)
#        input = input.view(-1, self.input_dim)
#        #input_recon, _ = self.forward(input, std=std)
#        input_recon = self.main(input)
#        var = std**2
#        grad = (input_recon - input) / var # -loglikelihood
#        return grad.view(sz)
#
#
#class WNConditionalDAE(nn.Module):
#    def __init__(self,
#                 input_dim=2, #10,
#                 h_dim=1000,
#                 context_dim=2,
#                 std=0.1,
#                 num_hidden_layers=1,
#                 nonlinearity='tanh',
#                 noise_type='gaussian',
#                 enc_input=False,
#                 enc_ctx=True,
#                 ):
#        super().__init__()
#        raise NotImplementedError
#        self.input_dim = input_dim
#        self.h_dim = h_dim
#        self.context_dim = context_dim
#        self.std = std
#        self.num_hidden_layers = num_hidden_layers
#        self.nonlinearity = nonlinearity
#        self.noise_type = noise_type
#        self.enc_input = enc_input
#        if self.enc_input:
#            inp_dim = h_dim
#        else:
#            inp_dim = input_dim
#        self.enc_ctx = enc_ctx
#        if enc_ctx:
#            ctx_dim = h_dim
#        else:
#            ctx_dim = context_dim
#
#        self.ctx_encode = Identity() if not self.enc_ctx \
#                else WNMLP(context_dim, h_dim, h_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers-1, use_nonlinearity_output=True, use_norm_output=True)
#        self.inp_encode = Identity() if not self.enc_input \
#                else WNMLP(input_dim,   h_dim, h_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers-1, use_nonlinearity_output=True, use_norm_output=True)
#        self.dae = WNMLP(inp_dim+h_dim, h_dim, input_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers, use_nonlinearity_output=False, use_norm_output=False)
#
#    def add_noise(self, input, std=None):
#        std = self.std if std is None else std
#        if self.noise_type == 'gaussian':
#            return add_gaussian_noise(input, std)
#        elif self.noise_type == 'uniform':
#            return add_uniform_noise(input, std)
#        else:
#            raise NotImplementedError
#
#    def loss(self, input, target):
#        # recon loss (likelihood)
#        recon_loss = F.mse_loss(input, target)#, reduction='sum')
#        return recon_loss
#
#    def forward(self, input, context, std=None):
#        # init
#        batch_size = input.size(0)
#        input = input.view(batch_size, self.input_dim)
#        context = context.view(batch_size, -1)
#
#        # add noise
#        x_bar = self.add_noise(input, std)
#
#        # encode
#        ctx = self.ctx_encode(context)
#        inp = self.inp_encode(x_bar)
#
#        # concat
#        h = torch.cat([inp, enc], dim=1)
#
#        # de-noise with context
#        x_recon = self.dae(h)
#
#        ''' get loss '''
#        loss = self.loss(x_recon, input)
#
#        # return
#        return x_recon, loss
#
#    def glogprob(self, input, context, std=None):
#        # init
#        sz = input.size()
#        batch_size = input.size(0)
#        input = input.view(batch_size, self.input_dim)
#        context = context.view(batch_size, -1)
#
#        # encode
#        ctx = self.ctx_encode(context)
#        inp = self.inp_encode(input)
#
#        # concat
#        h = torch.cat([inp, ctx], dim=1)
#
#        # de-noise with context
#        input_recon = self.dae(h)
#
#        # grad
#        var = std**2
#        grad = (input_recon - input) / var # -loglikelihood
#        return grad.view(sz)
#
#
#class ContextCtxInpConditionalDAE(nn.Module):
#    def __init__(self,
#                 input_dim=2, #10,
#                 h_dim=128,
#                 context_dim=2,
#                 std=0.1,
#                 num_hidden_layers=1,
#                 nonlinearity='tanh',
#                 noise_type='gaussian',
#                 enc_input=False,
#                 enc_ctx=True,
#                 init='gaussian',
#                 ):
#        super().__init__()
#        self.input_dim = input_dim
#        self.h_dim = h_dim
#        self.context_dim = context_dim
#        self.std = std
#        self.num_hidden_layers = num_hidden_layers
#        self.nonlinearity = nonlinearity
#        self.noise_type = noise_type
#        self.enc_input = enc_input
#        if self.enc_input:
#            inp_dim = h_dim
#        else:
#            inp_dim = input_dim
#        self.enc_ctx = enc_ctx
#        if enc_ctx:
#            ctx_dim = h_dim
#        else:
#            ctx_dim = context_dim
#        self.init = init
#
#        self.ctx_encode = Identity() if not self.enc_ctx \
#                else MLP(context_dim, h_dim, h_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers-1, use_nonlinearity_output=True)
#        self.inp_encode = Identity() if not self.enc_input \
#                else MLP(input_dim, h_dim, h_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers-1, use_nonlinearity_output=True)
#        self.dae = ContextScaleMLP(input_dim=ctx_dim, context_dim=inp_dim, hidden_dim=h_dim, output_dim=input_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers, use_nonlinearity_output=False)
#
#        if self.init == 'gaussian':
#            self.reset_parameters()
#        else:
#            pass
#
#    def reset_parameters(self):
#        nn.init.normal_(self.dae.fc.cbias.weight)
#        nn.init.normal_(self.dae.fc.direction)
#
#    def add_noise(self, input, std=None):
#        std = self.std if std is None else std
#        if self.noise_type == 'gaussian':
#            return add_gaussian_noise(input, std)
#        elif self.noise_type == 'uniform':
#            return add_uniform_noise(input, std)
#        else:
#            raise NotImplementedError
#
#    def loss(self, input, target):
#        # recon loss (likelihood)
#        recon_loss = F.mse_loss(input, target)#, reduction='sum')
#        return recon_loss
#
#    def forward(self, input, context, std=None):
#        # init
#        batch_size = input.size(0)
#        input = input.view(batch_size, self.input_dim)
#        context = context.view(batch_size, -1)
#
#        # add noise
#        x_bar = self.add_noise(input, std=std)
#
#        # encode
#        ctx = self.ctx_encode(context)
#        inp = self.inp_encode(x_bar)
#
#        # de-noise with context
#        x_recon = self.dae(ctx, inp)
#
#        ''' get loss '''
#        loss = self.loss(x_recon, input)
#
#        # return
#        return x_recon, loss
#
#    def glogprob(self, input, context, std=None):
#        # init
#        sz = input.size()
#        batch_size = input.size(0)
#        input = input.view(batch_size, self.input_dim)
#        context = context.view(batch_size, -1)
#
#        # encode
#        ctx = self.ctx_encode(context)
#        inp = self.inp_encode(input)
#
#        # de-noise with context
#        input_recon = self.dae(ctx, inp)
#
#        # grad
#        var = std**2
#        grad = (input_recon - input) / var # -loglikelihood
#        return grad.view(sz)
#
#class ContextInpCtxConditionalDAE(nn.Module):
#    def __init__(self,
#                 input_dim=2, #10,
#                 h_dim=128,
#                 context_dim=2,
#                 std=0.1,
#                 num_hidden_layers=1,
#                 nonlinearity='tanh',
#                 noise_type='gaussian',
#                 enc_input=False,
#                 enc_ctx=True,
#                 init='gaussian',
#                 ):
#        super().__init__()
#        self.input_dim = input_dim
#        self.h_dim = h_dim
#        self.context_dim = context_dim
#        self.std = std
#        self.num_hidden_layers = num_hidden_layers
#        self.nonlinearity = nonlinearity
#        self.noise_type = noise_type
#        self.enc_input = enc_input
#        if self.enc_input:
#            inp_dim = h_dim
#        else:
#            inp_dim = input_dim
#        self.enc_ctx = enc_ctx
#        if enc_ctx:
#            ctx_dim = h_dim
#        else:
#            ctx_dim = context_dim
#        self.init = init
#
#        self.ctx_encode = Identity() if not self.enc_ctx \
#                else MLP(context_dim, h_dim, h_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers-1, use_nonlinearity_output=True)
#        self.inp_encode = Identity() if not self.enc_input \
#                else MLP(input_dim, h_dim, h_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers-1, use_nonlinearity_output=True)
#        self.dae = ContextScaleMLP(input_dim=inp_dim, context_dim=ctx_dim, hidden_dim=h_dim, output_dim=input_dim, nonlinearity=nonlinearity, num_hidden_layers=num_hidden_layers, use_nonlinearity_output=False)
#
#        if self.init == 'gaussian':
#            self.reset_parameters()
#        else:
#            pass
#
#    def reset_parameters(self):
#        nn.init.normal_(self.dae.fc.cbias.weight)
#        nn.init.normal_(self.dae.fc.direction)
#
#    def add_noise(self, input, std=None):
#        std = self.std if std is None else std
#        if self.noise_type == 'gaussian':
#            return add_gaussian_noise(input, std)
#        elif self.noise_type == 'uniform':
#            return add_uniform_noise(input, std)
#        else:
#            raise NotImplementedError
#
#    def loss(self, input, target):
#        # recon loss (likelihood)
#        recon_loss = F.mse_loss(input, target)#, reduction='sum')
#        return recon_loss
#
#    def forward(self, input, context, std=None):
#        # init
#        batch_size = input.size(0)
#        input = input.view(batch_size, self.input_dim)
#        context = context.view(batch_size, -1)
#
#        # add noise
#        x_bar = self.add_noise(input, std=std)
#
#        # encode
#        ctx = self.ctx_encode(context)
#        inp = self.inp_encode(x_bar)
#
#        # de-noise with context
#        x_recon = self.dae(inp, ctx)
#
#        ''' get loss '''
#        loss = self.loss(x_recon, input)
#
#        # return
#        return x_recon, loss
#
#    def glogprob(self, input, context, std=None):
#        # init
#        sz = input.size()
#        batch_size = input.size(0)
#        input = input.view(batch_size, self.input_dim)
#        context = context.view(batch_size, -1)
#
#        # encode
#        ctx = self.ctx_encode(context)
#        inp = self.inp_encode(input)
#
#        # de-noise with context
#        input_recon = self.dae(inp, ctx)
#
#        # grad
#        var = std**2
#        grad = (input_recon - input) / var # -loglikelihood
#        return grad.view(sz)
