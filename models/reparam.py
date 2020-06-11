import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


MIN_LOGVAR = -4.
MAX_LOGVAR = 2.


''' NormalDistribution '''
class NormalDistribution(nn.Module):
    def __init__(self, nonlinearity=None):
        super(NormalDistribution, self).__init__()
        self.nonlinearity = nonlinearity

    def clip_logvar(self, logvar):
        # clip logvar values
        if self.nonlinearity == 'hard':
            logvar = torch.max(logvar, MIN_LOGVAR*torch.ones_like(logvar))
            logvar = torch.min(logvar, MAX_LOGVAR*torch.ones_like(logvar))
        elif self.nonlinearity == 'softplus':
            logvar = F.softplus(logvar)
        elif self.nonlinearity == 'spm10':
            logvar = F.softplus(logvar+10.) - 10.
        elif self.nonlinearity == 'spm6':
            logvar = F.softplus(logvar+6.) - 6.
        elif self.nonlinearity == 'spm5':
            logvar = F.softplus(logvar+5.) - 5.
        elif self.nonlinearity == 'spm4':
            logvar = F.softplus(logvar+4.) - 4.
        elif self.nonlinearity == 'spm3':
            logvar = F.softplus(logvar+3.) - 3.
        elif self.nonlinearity == 'spm2':
            logvar = F.softplus(logvar+2.) - 2.
        elif self.nonlinearity == 'tanh':
            logvar = F.tanh(logvar)
        elif self.nonlinearity == '2tanh':
            logvar = 2.0*F.tanh(logvar)
        return logvar

    def sample_gaussian(self, mu, logvar):
        #if self.training:
        #    std = torch.exp(0.5*logvar)
        #    eps = torch.randn_like(std)
        #    return mu + std * eps
        #else:
        #    return mu
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    #def forward(self, input):
    #    raise NotImplementedError()
    def forward(self, input):
        mu = self.mean_fn(input)
        logvar = self.clip_logvar(self.logvar_fn(input))
        return mu, logvar
        #output = self.sample_gaussian(mu, logvar)
        #return mu, logvar, output

class NormalDistributionLinear(NormalDistribution):
    def __init__(self, input_size, output_size, nonlinearity=None):
        super(NormalDistributionLinear, self).__init__(nonlinearity=nonlinearity)

        self.input_size = input_size
        self.output_size = output_size

        # define net
        self.mean_fn = nn.Linear(input_size, output_size)
        self.logvar_fn = nn.Linear(input_size, output_size)

    #def forward(self, input):
    #    mu = self.mean_fn(input)
    #    logvar = self.clip_logvar(self.logvar_fn(input))
    #    return mu, logvar

class NormalDistributionConv2d(NormalDistribution):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, nonlinearity=None):
        super(NormalDistributionConv2d, self).__init__(nonlinearity=nonlinearity)

        # define net
        self.mean_fn = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.logvar_fn = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    #def forward(self, input):
    #    mu = self.mean_fn(input)
    #    logvar = self.clip_logvar(self.logvar_fn(input))
    #    return mu, logvar

class NormalDistributionConvTranspose2d(NormalDistribution):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True, nonlinearity=None):
        super(NormalDistributionConvTranspose2d, self).__init__(nonlinearity=nonlinearity)

        # define net
        self.mean_fn = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias)
        self.logvar_fn = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias)

    #def forward(self, input):
    #    mu = self.mean_fn(input)
    #    logvar = self.clip_logvar(self.logvar_fn(input))
    #    return mu, logvar


''' BernoulliDistribution '''
class BernoulliDistribution(nn.Module):
    def __init__(self, hard=False):
        super(BernoulliDistribution, self).__init__()
        self.hard = hard

    def _sample_logistic(self, logits, eps=1e-20):
        ''' Sample from Logistic(0, 1) '''
        noise = torch.rand_like(logits)
        return torch.log(torch.div(noise, 1.-noise) + eps)

    def _sample_logistic_sigmoid(self, logits, temperature):
        ''' Draw a sample from the Logistic-Sigmoid distribution (Binary Concrete distribution) '''
        ''' See, https://arxiv.org/abs/1611.00712 ''' 
        y = logits + self._sample_logistic(logits)
        return torch.sigmoid(y / temperature)

    def sample_logistic_sigmoid(self, logits, temperature=1.0, hard=False):
        ''' Sample from the Logistic-Sigmoid distribution and optionally discretize.
        Args:
          logits: [batch_size, output_size] unnormalized log-probs
          temperature: non-negative scalar
          hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
          [batch_size, output_size] sample from the Logistic-Sigmoid distribution.
          If hard=True, then the returned sample will be one-hot, otherwise it will
          be a probabilitiy distribution that sums to 1 across classes
        ''' 
        y = self._sample_logistic_sigmoid(logits, temperature)

        if hard:
            raise NotImplementedError('current code for torch 0.1')
            # see https://github.com/yandexdataschool/gumbel_lstm/blob/master/gumbel_sigmoid.py
            #"""computes a hard indicator function. Not differentiable"""
            #y = T.switch(T.gt(logits,0),1,0)

            # check dimension
            assert y.dim() == 2

            # init y_hard
            if args.cuda:
                y_hard = torch.cuda.FloatTensor(logits.size()).zero_()
            else:
                y_hard = torch.FloatTensor(logits.size()).zero_()
            y_hard = Variable(y_hard)

            # get hard representation for y (into y_hard)
            y_hard.data[torch.gt(y.data, 0.5)] = 1.0

            # get hard
            y_hard.data.add(-1, y.data)
            y = y_hard + y

        return y

    def forward(self, input):
        raise NotImplementedError()

class BernoulliDistributionLinear(BernoulliDistribution):
    def __init__(self, input_size, output_size, hard=False):
        super(BernoulliDistributionLinear, self).__init__(hard=hard)
        self.input_size = input_size
        self.output_size = output_size

        # define net
        self.logit_fn = nn.Linear(input_size, output_size)

    def forward(self, input):
        logits = self.logit_fn(input)
        return logits
        #output = self.sample_logistic_sigmoid(logits, temperature, self.hard)
        #return logits, output

class BernoulliDistributionConv2d(BernoulliDistribution):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, hard=False):
        super(BernoulliDistributionConv2d, self).__init__(hard=hard)

        # define net
        self.logit_fn = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, input):
        logits = self.logit_fn(input)
        return logits
        #output = self.sample_logistic_sigmoid(logits, temperature, self.hard)
        #return logits, output

class BernoulliDistributionConvTranspose2d(BernoulliDistribution):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True, hard=False):
        super(BernoulliDistributionConvTranspose2d, self).__init__(hard=hard)

        # define net
        self.logit_fn = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias)

    def forward(self, input):
        logits = self.logit_fn(input)
        return logits
        #output = self.sample_logistic_sigmoid(logits, temperature, self.hard)
        #return logits, output


''' CategoricalDistribution '''
class CategoricalDistribution(nn.Module):
    def __init__(self, hard=False):
        super(CategoricalDistribution, self).__init__()
        self.hard = hard

    def _sample_gumbel(self, input, eps=1e-20):
        ''' Sample from Gumbel(0, 1) '''
        noise = torch.rand_like(input)
        return -torch.log(-torch.log(noise + eps) + eps)

    def _sample_gumbel_softmax(self, logits, temperature):
        ''' Draw a sample from the Gumbel-Softmax distribution '''
        y = logits + self._sample_gumbel(input)
        return self.softmax(y / temperature)

    def sample_gumbel_softmax(self, logits, temperature=1.0, hard=False):
        ''' Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
          logits: [batch_size, num_class] unnormalized log-probs
          temperature: non-negative scalar
          hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
          [batch_size, num_class] sample from the Gumbel-Softmax distribution.
          If hard=True, then the returned sample will be one-hot, otherwise it will
          be a probabilitiy distribution that sums to 1 across classes
        ''' 
        y = self._sample_gumbel_softmax(logits, temperature)

        if hard:
            raise NotImplementedError('current code for torch 0.1')
            #k = tf.shape(logits)[-1]
            ##y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
            #y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
            #y = tf.stop_gradient(y_hard - y) + y

            # check dimension
            assert y.dim() == 2

            # init y_hard
            if args.cuda:
                y_hard = torch.cuda.FloatTensor(logits.size()).zero_()
            else:
                y_hard = torch.FloatTensor(logits.size()).zero_()
            y_hard = Variable(y_hard)

            # get one-hot representation for y (into y_hard)
            val, ind = torch.max(y.data, 1)
            y_hard.data.scatter_(1, ind, 1)

            # get hard
            y_hard.data.add(-1, y.data)
            y = y_hard + y

        return y

    def forward(self, input):
        raise NotImplementedError()

class CategoricalDistributionLinear(CategoricalDistribution):
    def __init__(self, input_size, num_class, hard=False):
        super(CategoricalDistributionLinear, self).__init__(hard=hard)

        self.input_size = input_size
        self.num_class = num_class

        # define net
        self.logit_fn = nn.Linear(input_size, num_class)

    def forward(self, input):
        logits = self.logit_fn(input)
        return logits
        #output = self.gumbel_softmax(logits, temperature, self.hard)
        #return logits, output

class CategoricalDistributionConv2d(CategoricalDistribution):
    def __init__(self, in_channels, num_class, kernel_size, stride=1, padding=0, hard=False):
        super(CategoricalDistributionConv2d, self).__init__(hard=hard)
        self.num_class = num_class

        # define net
        self.logit_fn = nn.Conv2d(in_channels, num_class, kernel_size, stride, padding)

    def forward(self, input):
        logits = self.logit_fn(input)

        return logits
        #output = self.gumbel_softmax(logits, temperature, self.hard)
        #return logits, output

    def sample_gumbel_softmax(self, logits, temperature=1.0, hard=False):
        batch_size = logits.size(0)
        n_channels = logits.size(1)
        height = logits.size(2)
        width = logits.size(3)

        # 4d tensor [b c h w] to 2d tensor [bhw c]
        logits = torch.transpose(logits.view(batch_size, n_channels, height*width), 1, 2).view(batch_size*height*width, n_channels)

        # sample
        y = super(CategoricalDistributionConv2d, self).sample_gumbel_softmax(logits, temperature, hard)

        # 2d tensor [bhw c] to 4d tensor [b c h w]
        y = torch.transpose(y.view(batch_size, height*width, n_channels), 1, 2).view(batch_size, n_channels, height, width)
        return y
