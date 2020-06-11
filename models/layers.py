'''
copied and modified from https://github.com/CW-Huang/torchkit/blob/33f61b914bf8e79faebab3d3d64c17ea921ce6d2/torchkit/nn.py
copied and modified from https://github.com/lim0606/pytorch-flows-dev/blob/master/flows.py
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_nonlinear_func
from torch.nn.modules.utils import _pair

'''
miscellanious layers
'''
class Identity(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, input):
        return input

'''
copied and modified from https://github.com/CW-Huang/torchkit/blob/master/nn.py
'''
class WeightNormalizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, mask=None, norm=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('mask',mask)
        self.norm = norm
        self.direction = nn.Parameter(torch.Tensor(out_features, in_features))
        self.scale = nn.Parameter(torch.Tensor(out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.direction.size(1))
        self.direction.data.uniform_(-stdv, stdv)
        self.scale.data.uniform_(1, 1)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if self.norm:
            dir_ = self.direction
            direction = dir_.div(dir_.pow(2).sum(1).sqrt()[:,None])
            weight = self.scale[:,None].mul(direction)
        else:
            weight = self.scale[:,None].mul(self.direction)
        if self.mask is not None:
            #weight = weight * getattr(self.mask,⋅
            #                          ('cpu', 'cuda')[weight.is_cuda])()
            weight = weight * self.mask
        return F.linear(input, weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'


class ResLinear(nn.Module):
    def __init__(self,
                 in_features, out_features, bias=True, same_dim=False,
                 activation=nn.ReLU(), oper=WeightNormalizedLinear, oper_kwargs={'norm': False}):
        super().__init__()

        self.same_dim = same_dim

        self.dot_0h = oper(in_features, out_features, bias, **oper_kwargs)
        self.dot_h1 = oper(out_features, out_features, bias, **oper_kwargs)
        if not same_dim:
            self.dot_01 = oper(in_features, out_features, bias, **oper_kwargs)

        self.activation = activation

    def forward(self, input):
        h = self.activation(self.dot_0h(input))
        out_nonlinear = self.dot_h1(h)
        out_skip = input if self.same_dim else self.dot_01(input)
        return out_nonlinear + out_skip

class ContextResLinear(nn.Module):
    def __init__(self,
                 in_features, out_features, context_features, bias=True, same_dim=False,
                 activation=nn.ReLU(), oper=WeightNormalizedLinear, oper_kwargs={'norm': False}):
        super().__init__()

        self.same_dim = same_dim

        self.dot_0h = oper(in_features, out_features, bias, **oper_kwargs)
        self.dot_h1 = oper(out_features, out_features, bias, **oper_kwargs)
        if not same_dim:
            self.dot_01 = oper(in_features, out_features, bias, **oper_kwargs)

        self.dot_0c = oper(context_features, out_features, bias, **oper_kwargs)
        self.dot_c1 = oper(out_features, out_features, bias, **oper_kwargs)

        self.activation = activation

    def forward(self, input, context):
        h = self.activation(self.dot_0h(input))
        outi_nonlinear = self.dot_h1(h)
        c = self.activation(self.dot_0c(context))
        outc_nonlinear = self.dot_c1(c)
        out_skip = input if self.same_dim else self.dot_01(input)
        return outi_nonlinear + outc_nonlinear + out_skip


''' context '''
class ContextLinear(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']
    def __init__(self, in_features, out_features, context_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.context_features = context_features
        self.direction = nn.Parameter(torch.Tensor(out_features, in_features))
        self.cscale = nn.Linear(context_features, out_features, bias=False)
        self.cbias = nn.Linear(context_features, out_features, bias=bias)
        #self.cbias = nn.Linear(in_features+context_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.direction, a=math.sqrt(5))
        self.cscale.weight.data.normal_(0, 0.005)
        #torch.nn.init.constant_(self.cscale.bias, 1)
        #self.cbias.weight.data.normal_(0, 0.001)
        #torch.nn.init.constant_(self.cbias.bias, 0)

    def forward(self, input, context):
        scale = 1.+self.cscale(context)
        bias = self.cbias(context)
        return scale * F.linear(input, self.direction, None) + bias
        #return scale * self.cbias(torch.cat([input, context], dim=1))

    def extra_repr(self):
        return 'in_features={}, out_features={}, context_features={}'.format(
            self.in_features, self.out_features, self.context_features,
        )

class ContextConv2d(nn.Module):
    def __init__(self,
                 in_channels, out_channels, context_channels,
                 kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.context_channels = context_channels
        self.direction = nn.Conv2d(in_channels, out_channels, bias=False, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)#, padding_mode=padding_mode)
        self.cscale = nn.Conv2d(context_channels, out_channels, bias=False, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)#, padding_mode=padding_mode)
        self.cbias = nn.Conv2d(context_channels, out_channels, bias=bias, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)#, padding_mode=padding_mode)
        self.reset_parameters()

    def reset_parameters(self):
        #torch.nn.init.kaiming_uniform_(self.direction, a=math.sqrt(5))
        self.cscale.weight.data.normal_(0, 0.005)
        #torch.nn.init.constant_(self.cscale.bias, 1)
        #self.cbias.weight.data.normal_(0, 0.001)
        #torch.nn.init.constant_(self.cbias.bias, 0)

    def forward(self, input, context):
        scale = 1.+self.cscale(context)
        bias = self.cbias(context)
        return scale * self.direction(input) + bias

    def extra_repr(self):
        return 'in_channels={}, out_channels={}, context_channels={}'.format(
            self.in_channels, self.out_channels, self.context_channels,
        )

class ContextWeightNormalizedLinear(nn.Module):
    def __init__(self, in_features, out_features, context_features, bias=True, in_norm=False, ctx_norm=True, ctx_scale=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.context_features = context_features
        self.in_norm = in_norm
        self.ctx_norm = ctx_norm
        self.ctx_scale = ctx_scale
        self.direction = nn.Parameter(torch.Tensor(out_features, in_features))
        self.cscale = nn.Parameter(torch.Tensor(out_features, context_features))
        self.cbias = nn.Linear(context_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.direction, a=math.sqrt(5))
        self.cscale.data.normal_(0, 0.005)
        #self.cscale.weight.data.normal_(0, 0.1)
        #self.cbias.weight.data.normal_(0, 0.1)
        #self.direction.data.normal_(0, 0.001)

    def forward(self, input, context):
        bias = self.cbias(context)
        if self.ctx_norm:
            cscale_ = self.cscale
            cscale = cscale_.div(cscale_.pow(2).sum(1).sqrt()[:,None])
            scale = 1.+self.ctx_scale*F.linear(context, cscale, None)
        else:
            scale = 1.+F.linear(context, self.cscale, None)
        if self.in_norm:
            dir_ = self.direction
            weight = dir_.div(dir_.pow(2).sum(1).sqrt()[:,None])
        else:
            weight = self.direction
        return scale * F.linear(input, weight, None) + bias

    def extra_repr(self):
        return 'in_features={}, out_features={}, context_features={}, in_norm={}, ctx_norm={}'.format(
            self.in_features, self.out_features, self.context_features, self.in_norm, self.ctx_norm
        )


''' context (softplus) '''
class ContextSoftPlusLinear(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']
    def __init__(self, in_features, out_features, context_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.context_features = context_features
        self.direction = nn.Parameter(torch.Tensor(out_features, in_features))
        self.cscale = nn.Linear(context_features, out_features, bias=True)
        self.cbias = nn.Linear(context_features, out_features, bias=bias)
        #self.cbias = nn.Linear(in_features+context_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.direction, a=math.sqrt(5))
        self.cscale.weight.data.normal_(0, 0.005)
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.cscale.weight)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.cscale.bias, -bound, bound)
        #torch.nn.init.constant_(self.cscale.bias, 1)
        #self.cbias.weight.data.normal_(0, 0.001)
        #torch.nn.init.constant_(self.cbias.bias, 0)

    def forward(self, input, context):
        scale = F.softplus(self.cscale(context))
        bias = self.cbias(context)
        return scale * F.linear(input, self.direction, None) + bias
        #return scale * self.cbias(torch.cat([input, context], dim=1))

    def extra_repr(self):
        return 'in_features={}, out_features={}, context_features={}'.format(
            self.in_features, self.out_features, self.context_features,
        )

class ContextSoftPlusConv2d(nn.Module):
    def __init__(self,
                 in_channels, out_channels, context_channels,
                 kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.context_channels = context_channels
        self.direction = nn.Conv2d(in_channels, out_channels, bias=False, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)#, padding_mode=padding_mode)
        self.cscale = nn.Conv2d(context_channels, out_channels, bias=True, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)#, padding_mode=padding_mode)
        self.cbias = nn.Conv2d(context_channels, out_channels, bias=bias, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)#, padding_mode=padding_mode)
        self.reset_parameters()

    def reset_parameters(self):
        #torch.nn.init.kaiming_uniform_(self.direction, a=math.sqrt(5))
        self.cscale.weight.data.normal_(0, 0.005)
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.cscale.weight)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.cscale.bias, -bound, bound)
        #torch.nn.init.constant_(self.cscale.bias, 1)
        #self.cbias.weight.data.normal_(0, 0.001)
        #torch.nn.init.constant_(self.cbias.bias, 0)

    def forward(self, input, context):
        scale = F.softplus(self.cscale(context))
        bias = self.cbias(context)
        return scale * self.direction(input) + bias

    def extra_repr(self):
        return 'in_channels={}, out_channels={}, context_channels={}'.format(
            self.in_channels, self.out_channels, self.context_channels,
        )

class ContextSoftPlusWeightNormalizedLinear(nn.Module):
    def __init__(self, in_features, out_features, context_features, bias=True, in_norm=False, ctx_norm=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.context_features = context_features
        self.in_norm = in_norm
        self.ctx_norm = ctx_norm
        self.direction = nn.Parameter(torch.Tensor(out_features, in_features))
        self.cscale = nn.Parameter(torch.Tensor(out_features, context_features))
        self.cscalebias = nn.Parameter(torch.Tensor(out_features))
        self.cbias = nn.Linear(context_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.direction, a=math.sqrt(5))
        self.cscale.data.normal_(0, 1)
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.cscale)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.cscalebias, -bound, bound)
        #self.cscale.weight.data.normal_(0, 0.1)
        #self.cbias.weight.data.normal_(0, 0.1)
        #self.direction.data.normal_(0, 0.001)

    def forward(self, input, context):
        bias = self.cbias(context)
        if self.ctx_norm:
            cscale_ = self.cscale
            cscale = cscale_.div(cscale_.pow(2).sum(1).sqrt()[:,None])
            scale = F.softplus(F.linear(context, cscale, self.cscalebias))
        else:
            scale = F.softplus(F.linear(context, self.cscale, self.cscalebias))
        if self.in_norm:
            dir_ = self.direction
            weight = dir_.div(dir_.pow(2).sum(1).sqrt()[:,None])
        else:
            weight = self.direction
        return scale * F.linear(input, weight, None) + bias

    def extra_repr(self):
        return 'in_features={}, out_features={}, context_features={}, in_norm={}, ctx_norm={}'.format(
            self.in_features, self.out_features, self.context_features, self.in_norm, self.ctx_norm
        )

class ContextSoftPlusWeightNormalizedConv2d(nn.Module):
    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                     #'padding_mode', 'output_padding',
                     'in_channels', 'out_channels', 'context_channels', 'kernel_size']
    def __init__(self,
                 in_channels, out_channels, context_channels,
                 kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, in_norm=False, ctx_norm=True):#padding_mode='zeros'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.context_channels = context_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.in_norm = in_norm
        self.ctx_norm = ctx_norm
        self.direction = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.cscale = nn.Parameter(torch.Tensor(out_channels, context_channels, kernel_size, kernel_size))
        self.cscalebias = nn.Parameter(torch.Tensor(out_channels))
        self.cbias = nn.Conv2d(context_channels, out_channels, bias=bias, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)#, padding_mode=padding_mode)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.direction, a=math.sqrt(5))
        self.cscale.data.normal_(0, 1)
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.cscale)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.cscalebias, -bound, bound)
        #torch.nn.init.constant_(self.cscale.bias, 1)
        #self.cbias.weight.data.normal_(0, 0.001)
        #torch.nn.init.constant_(self.cbias.bias, 0)

    def forward(self, input, context):
        bias = self.cbias(context)
        if self.ctx_norm:
            cscale_ = self.cscale
            cscale = cscale_.div(cscale_.pow(2).sum(1).sum(1).sum(1).sqrt()[:,None,None,None])
            scale = F.softplus(F.conv2d(context, cscale, bias=self.cscalebias,
                stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups))
        else:
            scale = F.softplus(F.conv2d(context, self.cscale, bias=self.cscalebias,
                stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups))
        if self.in_norm:
            dir_ = self.direction
            weight = dir_.div(dir_.pow(2).sum(1).sum(1).sum(1).sqrt()[:,None,None,None])
        else:
            weight = self.direction
        out = F.conv2d(input, weight, bias=None,
                stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return scale * out + bias

    def extra_repr(self):
            s = ('{in_channels}, {out_channels}, {context_channels}, in_norm={in_norm}, ctx_norm={ctx_norm}, kernel_size={kernel_size}'
                 ', stride={stride}')
            if self.padding != 0:
                s += ', padding={padding}'
            if self.dilation != 1:
                s += ', dilation={dilation}'
            if self.groups != 1:
                s += ', groups={groups}'
            if self.bias is None:
                s += ', bias=False'
            return s.format(**self.__dict__)

''' bilinear '''
class SimplifiedBilinear(nn.Module):
    def __init__(self, in1_features, in2_features, out_features, bias=True):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.path1 = nn.Linear(in1_features, out_features, bias=bias)
        self.path2 = nn.Linear(in2_features, out_features, bias=False)

    def forward(self, input1, input2):
        return self.path1(input1) + self.path2(input2)

    def extra_repr(self):
        return 'in1_features={}, in2_features={}, out_features={}'.format(
            self.in1_features, self.in2_features, self.out_features,
        )

class WeightNormalizedSimplifiedBilinear(nn.Module):
    def __init__(self, in1_features, in2_features, out_features, bias=True, in1_norm=False, in2_norm=True):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.in1_norm = in1_norm
        self.in2_norm = in2_norm
        self.path1 = nn.Parameter(torch.Tensor(out_features, in1_features))
        self.path2 = nn.Parameter(torch.Tensor(out_features, in2_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.path1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.path2, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.path1)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input1, input2):
        if self.in1_norm:
            dir1_ = self.path1
            weight1 = dir1_.div(dir1_.pow(2).sum(1).sqrt()[:,None])
        else:
            weight1 = self.path1
        if self.in2_norm:
            dir2_ = self.path2
            weight2 = dir2_.div(dir2_.pow(2).sum(1).sqrt()[:,None])
        else:
            weight2 = self.path2
        return F.linear(input1, weight1, self.bias) + F.linear(input2, weight2, None)

    def extra_repr(self):
        return 'in1_features={}, in2_features={}, out_features={}, in1_norm={}, in2_norm={}'.format(
            self.in1_features, self.in2_features, self.out_features, self.in1_norm, self.in2_norm
        )

class StackedWeightNormalizedSimplifiedBilinear(nn.Module):
    def __init__(self, in1_features, in2_features, hid_features, out_features, bias=True, norm=True, nonlinearity='relu'):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.hid_features = hid_features
        self.out_features = out_features
        self.norm = norm
        self.nonlinearity = nonlinearity
        self.main = WeightNormalizedSimplifiedBilinear(in1_features, in2_features, hid_features, bias=bias, norm=norm)
        self.fc = nn.Linear(hid_features, out_features)

    def forward(self, input1, input2):
        afunc = get_nonlinear_func(self.nonlinearity)
        hid = afunc(self.main(input1, input2))
        out = self.fc(hid)
        return out


''' MLP '''
class MLP(nn.Module):
    def __init__(self,
                 input_dim=2,
                 hidden_dim=8,
                 output_dim=2,
                 nonlinearity='relu',
                 num_hidden_layers=1,
                 use_nonlinearity_output=False,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nonlinearity = nonlinearity
        self.num_hidden_layers = num_hidden_layers
        self.use_nonlinearity_output = use_nonlinearity_output

        layers = []
        if num_hidden_layers >= 1:
            for i in range(num_hidden_layers):
                layers += [nn.Linear(input_dim if i==0 else hidden_dim, hidden_dim)]
        self.layers = nn.ModuleList(layers)
        self.fc = nn.Linear(input_dim if num_hidden_layers==0 else hidden_dim, output_dim)

    def forward(self, input):
        # init
        batch_size = input.size(0)
        x = input.view(batch_size, self.input_dim)

        # forward
        hidden = x
        if self.num_hidden_layers >= 1:
            for i in range(self.num_hidden_layers):
                hidden = get_nonlinear_func(self.nonlinearity)(self.layers[i](hidden))
        output = self.fc(hidden)
        if self.use_nonlinearity_output:
            output = get_nonlinear_func(self.nonlinearity)(output)

        return output

class WNMLP(nn.Module):
    def __init__(self,
                 input_dim=2,
                 hidden_dim=8,
                 output_dim=2,
                 nonlinearity='relu',
                 num_hidden_layers=1,
                 use_nonlinearity_output=False,
                 use_norm_output=False,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nonlinearity = nonlinearity
        self.num_hidden_layers = num_hidden_layers
        self.use_nonlinearity_output = use_nonlinearity_output
        self.use_norm_output = use_norm_output

        layers = []
        if num_hidden_layers >= 1:
            for i in range(num_hidden_layers):
                layers += [WeightNormalizedLinear(input_dim if i==0 else hidden_dim, hidden_dim)]
        self.layers = nn.ModuleList(layers)
        self.fc = WeightNormalizedLinear(input_dim if num_hidden_layers==0 else hidden_dim, output_dim, norm=use_norm_output)

    def forward(self, input):
        # init
        batch_size = input.size(0)
        x = input.view(batch_size, self.input_dim)

        # forward
        hidden = x
        if self.num_hidden_layers >= 1:
            for i in range(self.num_hidden_layers):
                hidden = get_nonlinear_func(self.nonlinearity)(self.layers[i](hidden))
        output = self.fc(hidden)
        if self.use_nonlinearity_output:
            output = get_nonlinear_func(self.nonlinearity)(output)

        return output

class ResMLP(nn.Module):
    def __init__(self,
                 input_dim=2,
                 hidden_dim=8,
                 output_dim=2,
                 nonlinearity='relu',
                 num_hidden_layers=1,
                 use_nonlinearity_output=False,
                 layer='wnlinear',
                 use_norm=False,
                 use_norm_output=False,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nonlinearity = nonlinearity
        self.num_hidden_layers = num_hidden_layers
        self.use_nonlinearity_output = use_nonlinearity_output
        self.layer = layer
        self.use_norm = use_norm
        self.use_norm_output = use_norm_output

        if self.layer == 'linear':
            oper = nn.Linear
            oper_kwargs={}
        elif self.layer == 'wnlinear':
            oper = WeightNormalizedLinear
            oper_kwargs={'norm': use_norm}
        else:
            raise NotImplementedError

        layers = []
        prev_hidden_dim = input_dim
        if num_hidden_layers >= 1:
            for i in range(num_hidden_layers):
                layers += [ResLinear(input_dim if i==0 else hidden_dim,
                                     hidden_dim,
                                     same_dim=prev_hidden_dim==hidden_dim,
                                     oper=oper,
                                     oper_kwargs=oper_kwargs)]
                prev_hidden_dim = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.fc = ResLinear(input_dim if num_hidden_layers==0 else hidden_dim,
                            output_dim,
                            same_dim=prev_hidden_dim==output_dim,
                            oper=oper,
                            oper_kwargs=oper_kwargs)

    def forward(self, input):
        # init
        batch_size = input.size(0)
        x = input.view(batch_size, self.input_dim)

        # forward
        hidden = x
        if self.num_hidden_layers >= 1:
            for i in range(self.num_hidden_layers):
                hidden = get_nonlinear_func(self.nonlinearity)(self.layers[i](hidden))
        output = self.fc(hidden)
        if self.use_nonlinearity_output:
            output = get_nonlinear_func(self.nonlinearity)(output)

        return output

class ContextResMLP(nn.Module):
    def __init__(self,
                 input_dim=2,
                 context_dim=2,
                 hidden_dim=8,
                 output_dim=2,
                 nonlinearity='relu',
                 num_hidden_layers=1,
                 use_nonlinearity_output=False,
                 use_norm=False,
                 use_norm_output=False,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nonlinearity = nonlinearity
        self.num_hidden_layers = num_hidden_layers
        self.use_nonlinearity_output = use_nonlinearity_output
        self.use_norm = use_norm
        self.use_norm_output = use_norm_output

        layers = []
        prev_hidden_dim = input_dim
        if num_hidden_layers >= 1:
            for i in range(num_hidden_layers):
                layers += [ContextResLinear(input_dim if i==0 else hidden_dim,
                                     hidden_dim,
                                     context_dim,
                                     same_dim=prev_hidden_dim==hidden_dim,
                                     oper_kwargs={'norm': use_norm})]
                prev_hidden_dim = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.fc = ContextResLinear(input_dim if num_hidden_layers==0 else hidden_dim,
                            output_dim,
                            context_dim,
                            same_dim=prev_hidden_dim==output_dim,
                            oper_kwargs={'norm': use_norm_output})

    def forward(self, input, context):
        # init
        batch_size = input.size(0)
        x = input.view(batch_size, self.input_dim)
        ctx = context.view(batch_size, self.context_dim)

        # forward
        hidden = x
        if self.num_hidden_layers >= 1:
            for i in range(self.num_hidden_layers):
                hidden = get_nonlinear_func(self.nonlinearity)(self.layers[i](hidden, ctx))
        output = self.fc(hidden, ctx)
        if self.use_nonlinearity_output:
            output = get_nonlinear_func(self.nonlinearity)(output)

        return output

class ContextConcatMLP(nn.Module):
    def __init__(self,
                 input_dim=2,
                 context_dim=2,
                 hidden_dim=8,
                 output_dim=2,
                 nonlinearity='relu',
                 num_hidden_layers=1,
                 use_nonlinearity_output=False,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nonlinearity = nonlinearity
        self.num_hidden_layers = num_hidden_layers
        self.use_nonlinearity_output = use_nonlinearity_output

        layers = []
        if num_hidden_layers >= 1:
            for i in range(num_hidden_layers):
                layers += [nn.Linear(input_dim+context_dim if i==0 else hidden_dim+context_dim, hidden_dim)]
        self.layers = nn.ModuleList(layers)
        self.fc = nn.Linear(input_dim+context_dim if num_hidden_layers==0 else hidden_dim+context_dim, output_dim)

    def forward(self, input, context):
        # init
        batch_size = input.size(0)
        x = input.view(batch_size, self.input_dim)
        ctx = context.view(batch_size, self.context_dim)

        # forward
        hidden = x
        if self.num_hidden_layers >= 1:
            for i in range(self.num_hidden_layers):
                _hidden = torch.cat([hidden, ctx], dim=1)
                hidden = get_nonlinear_func(self.nonlinearity)(self.layers[i](_hidden))
        _hidden = torch.cat([hidden, ctx], dim=1)
        output = self.fc(_hidden)
        if self.use_nonlinearity_output:
            output = get_nonlinear_func(self.nonlinearity)(output)

        return output

class ContextScaleMLP(nn.Module):
    def __init__(self,
                 input_dim=2,
                 context_dim=2,
                 hidden_dim=8,
                 output_dim=2,
                 nonlinearity='relu',
                 num_hidden_layers=3,
                 use_nonlinearity_output=False,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nonlinearity = nonlinearity
        self.num_hidden_layers = num_hidden_layers
        self.use_nonlinearity_output = use_nonlinearity_output

        layers = []
        if num_hidden_layers >= 1:
            for i in range(num_hidden_layers):
                layers += [ContextLinear(
                    in_features=input_dim if i==0 else hidden_dim,
                    out_features=hidden_dim,
                    context_features=context_dim)]
        self.layers = nn.ModuleList(layers)
        self.fc = ContextLinear(
                in_features=input_dim if num_hidden_layers==0 else hidden_dim,
                out_features=output_dim,
                context_features=context_dim)

    #def reset_parameters(self):
    #    for layer in self.layers:
    #        layer.reset_parameters()
    #    self.fc.reset_parameters()

    def forward(self, input, context):
        # init
        batch_size = input.size(0)
        x = input.view(batch_size, self.input_dim)
        ctx = context.view(batch_size, self.context_dim)

        # forward
        hidden = x
        if self.num_hidden_layers >= 1:
            for i in range(self.num_hidden_layers):
                hidden = get_nonlinear_func(self.nonlinearity)(self.layers[i](hidden, ctx))
        output = self.fc(hidden, ctx)
        if self.use_nonlinearity_output:
            output = get_nonlinear_func(self.nonlinearity)(output)

        return output

class ContextWNScaleMLP(nn.Module):
    def __init__(self,
                 input_dim=2,
                 context_dim=2,
                 hidden_dim=8,
                 output_dim=2,
                 nonlinearity='relu',
                 num_hidden_layers=3,
                 use_nonlinearity_output=False,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nonlinearity = nonlinearity
        self.num_hidden_layers = num_hidden_layers
        self.use_nonlinearity_output = use_nonlinearity_output

        layers = []
        if num_hidden_layers >= 1:
            for i in range(num_hidden_layers):
                layers += [ContextWeightNormalizedLinear(
                    in_features=input_dim if i==0 else hidden_dim,
                    out_features=hidden_dim,
                    context_features=context_dim)]
        self.layers = nn.ModuleList(layers)
        self.fc = ContextWeightNormalizedLinear(
                in_features=input_dim if num_hidden_layers==0 else hidden_dim,
                out_features=output_dim,
                context_features=context_dim)

    #def reset_parameters(self):
    #    for layer in self.layers:
    #        layer.reset_parameters()
    #    self.fc.reset_parameters()

    def forward(self, input, context):
        # init
        batch_size = input.size(0)
        x = input.view(batch_size, self.input_dim)
        ctx = context.view(batch_size, self.context_dim)

        # forward
        hidden = x
        if self.num_hidden_layers >= 1:
            for i in range(self.num_hidden_layers):
                hidden = get_nonlinear_func(self.nonlinearity)(self.layers[i](hidden, ctx))
        output = self.fc(hidden, ctx)
        if self.use_nonlinearity_output:
            output = get_nonlinear_func(self.nonlinearity)(output)

        return output

class ContextSPScaleMLP(nn.Module):
    def __init__(self,
                 input_dim=2,
                 context_dim=2,
                 hidden_dim=8,
                 output_dim=2,
                 nonlinearity='relu',
                 num_hidden_layers=3,
                 use_nonlinearity_output=False,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nonlinearity = nonlinearity
        self.num_hidden_layers = num_hidden_layers
        self.use_nonlinearity_output = use_nonlinearity_output

        layers = []
        if num_hidden_layers >= 1:
            for i in range(num_hidden_layers):
                layers += [ContextSoftPlusLinear(
                    in_features=input_dim if i==0 else hidden_dim,
                    out_features=hidden_dim,
                    context_features=context_dim)]
        self.layers = nn.ModuleList(layers)
        self.fc = ContextSoftPlusLinear(
                in_features=input_dim if num_hidden_layers==0 else hidden_dim,
                out_features=output_dim,
                context_features=context_dim)

    def forward(self, input, context):
        # init
        batch_size = input.size(0)
        x = input.view(batch_size, self.input_dim)
        ctx = context.view(batch_size, self.context_dim)

        # forward
        hidden = x
        if self.num_hidden_layers >= 1:
            for i in range(self.num_hidden_layers):
                hidden = get_nonlinear_func(self.nonlinearity)(self.layers[i](hidden, ctx))
        output = self.fc(hidden, ctx)
        if self.use_nonlinearity_output:
            output = get_nonlinear_func(self.nonlinearity)(output)

        return output

class ContextSPWNScaleMLP(nn.Module):
    def __init__(self,
                 input_dim=2,
                 context_dim=2,
                 hidden_dim=8,
                 output_dim=2,
                 nonlinearity='relu',
                 num_hidden_layers=3,
                 use_nonlinearity_output=False,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nonlinearity = nonlinearity
        self.num_hidden_layers = num_hidden_layers
        self.use_nonlinearity_output = use_nonlinearity_output

        layers = []
        if num_hidden_layers >= 1:
            for i in range(num_hidden_layers):
                layers += [ContextSoftPlusWeightNormalizedLinear(
                    in_features=input_dim if i==0 else hidden_dim,
                    out_features=hidden_dim,
                    context_features=context_dim)]
        self.layers = nn.ModuleList(layers)
        self.fc = ContextSoftPlusWeightNormalizedLinear(
                in_features=input_dim if num_hidden_layers==0 else hidden_dim,
                out_features=output_dim,
                context_features=context_dim)

    def forward(self, input, context):
        # init
        batch_size = input.size(0)
        x = input.view(batch_size, self.input_dim)
        ctx = context.view(batch_size, self.context_dim)

        # forward
        hidden = x
        if self.num_hidden_layers >= 1:
            for i in range(self.num_hidden_layers):
                hidden = get_nonlinear_func(self.nonlinearity)(self.layers[i](hidden, ctx))
        output = self.fc(hidden, ctx)
        if self.use_nonlinearity_output:
            output = get_nonlinear_func(self.nonlinearity)(output)

        return output

class ContextBilinearMLP(nn.Module):
    def __init__(self,
                 input_dim=2,
                 context_dim=2,
                 hidden_dim=8,
                 output_dim=2,
                 nonlinearity='relu',
                 num_hidden_layers=3,
                 use_nonlinearity_output=False,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nonlinearity = nonlinearity
        self.num_hidden_layers = num_hidden_layers
        self.use_nonlinearity_output = use_nonlinearity_output

        layers = []
        if num_hidden_layers >= 1:
            for i in range(num_hidden_layers):
                layers += [SimplifiedBilinear(
                    in1_features=input_dim if i==0 else hidden_dim,
                    in2_features=context_dim,
                    out_features=hidden_dim,
                    )]
        self.layers = nn.ModuleList(layers)
        self.fc = SimplifiedBilinear(
                in1_features=input_dim if num_hidden_layers==0 else hidden_dim,
                in2_features=context_dim,
                out_features=output_dim,
                )

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.fc.reset_parameters()

    def forward(self, input, context):
        # init
        batch_size = input.size(0)
        x = input.view(batch_size, self.input_dim)
        ctx = context.view(batch_size, self.context_dim)

        # forward
        hidden = x
        if self.num_hidden_layers >= 1:
            for i in range(self.num_hidden_layers):
                hidden = get_nonlinear_func(self.nonlinearity)(self.layers[i](hidden, ctx))
        output = self.fc(hidden, ctx)
        if self.use_nonlinearity_output:
            output = get_nonlinear_func(self.nonlinearity)(output)

        return output

class ContextWNBilinearMLP(nn.Module):
    def __init__(self,
                 input_dim=2,
                 context_dim=2,
                 hidden_dim=8,
                 output_dim=2,
                 nonlinearity='relu',
                 num_hidden_layers=3,
                 use_nonlinearity_output=False,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nonlinearity = nonlinearity
        self.num_hidden_layers = num_hidden_layers
        self.use_nonlinearity_output = use_nonlinearity_output

        layers = []
        if num_hidden_layers >= 1:
            for i in range(num_hidden_layers):
                layers += [WeightNormalizedSimplifiedBilinear(
                    in1_features=input_dim if i==0 else hidden_dim,
                    in2_features=context_dim,
                    out_features=hidden_dim,
                    )]
        self.layers = nn.ModuleList(layers)
        self.fc = WeightNormalizedSimplifiedBilinear(
                in1_features=input_dim if num_hidden_layers==0 else hidden_dim,
                in2_features=context_dim,
                out_features=output_dim,
                )

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.fc.reset_parameters()

    def forward(self, input, context):
        # init
        batch_size = input.size(0)
        x = input.view(batch_size, self.input_dim)
        ctx = context.view(batch_size, self.context_dim)

        # forward
        hidden = x
        if self.num_hidden_layers >= 1:
            for i in range(self.num_hidden_layers):
                hidden = get_nonlinear_func(self.nonlinearity)(self.layers[i](hidden, ctx))
        output = self.fc(hidden, ctx)
        if self.use_nonlinearity_output:
            output = get_nonlinear_func(self.nonlinearity)(output)

        return output

class ContextSWNBilinearMLP(nn.Module):
    def __init__(self,
                 input_dim=2,
                 context_dim=2,
                 hidden_dim=8,
                 output_dim=2,
                 nonlinearity='relu',
                 num_hidden_layers=3,
                 use_nonlinearity_output=False,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nonlinearity = nonlinearity
        self.num_hidden_layers = num_hidden_layers
        self.use_nonlinearity_output = use_nonlinearity_output

        layers = []
        if num_hidden_layers >= 1:
            for i in range(num_hidden_layers):
                layers += [StackedWeightNormalizedSimplifiedBilinear(
                    in1_features=input_dim if i==0 else hidden_dim,
                    in2_features=context_dim,
                    hid_features=hidden_dim,
                    out_features=hidden_dim,
                    )]
        self.layers = nn.ModuleList(layers)
        self.fc = StackedWeightNormalizedSimplifiedBilinear(
                in1_features=input_dim if num_hidden_layers==0 else hidden_dim,
                in2_features=context_dim,
                hid_features=hidden_dim,
                out_features=output_dim,
                )

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.fc.reset_parameters()

    def forward(self, input, context):
        # init
        batch_size = input.size(0)
        x = input.view(batch_size, self.input_dim)
        ctx = context.view(batch_size, self.context_dim)

        # forward
        hidden = x
        if self.num_hidden_layers >= 1:
            for i in range(self.num_hidden_layers):
                hidden = get_nonlinear_func(self.nonlinearity)(self.layers[i](hidden, ctx))
        output = self.fc(hidden, ctx)
        if self.use_nonlinearity_output:
            output = get_nonlinear_func(self.nonlinearity)(output)

        return output
