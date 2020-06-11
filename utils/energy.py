import math
import torch


EPS = 1e-9

def regularization_func(x):
    return (torch.relu(x.abs() - 6) ** 2).sum(-1, keepdim=True)

def w1(z1):
    return torch.sin(2.*math.pi*z1/4.)

def w2(z1):
    return 3.*torch.exp(-0.5*((z1-1)/0.6)**2)

def w3(z1):
    return 3.*torch.sigmoid((z1-1.)/0.3)

def energy_func1(x):
    assert x.dim() == 2
    assert x.size(1) == 2
    batch_size = x.size(0)
    x1 = x[:, :1]
    x2 = x[:, 1:]
    xnorm = torch.norm(x, dim=1, keepdim=True)
    energy = 0.5 * ((xnorm-2)/0.4)**2 \
            - torch.log(
                    torch.exp(-0.5*((x1-2)/0.6)**2)
                    + torch.exp(-0.5*((x1+2)/0.6)**2)
                    + EPS)
    return energy + regularization_func(x)

def energy_func2(x):
    assert x.dim() == 2
    assert x.size(1) == 2
    batch_size = x.size(0)
    x1 = x[:, :1]
    x2 = x[:, 1:]

    energy = 0.5 * ((x2-w1(x1))/0.4)**2
    return energy + regularization_func(x)

def energy_func3(x):
    assert x.dim() == 2
    assert x.size(1) == 2
    batch_size = x.size(0)
    x1 = x[:, :1]
    x2 = x[:, 1:]

    energy = -torch.log(
            torch.exp(-0.5 * ((x2-w1(x1))/0.35)**2)
            + torch.exp(-0.5 * ((x2-w1(x1)+w2(x1))/0.35)**2)
            + EPS)
    return energy + regularization_func(x)

def energy_func4(x):
    assert x.dim() == 2
    assert x.size(1) == 2
    batch_size = x.size(0)
    x1 = x[:, :1]
    x2 = x[:, 1:]

    energy = -torch.log(
            torch.exp(-0.5 * ((x2-w1(x1))/0.4)**2)
            + torch.exp(-0.5 * ((x2-w1(x1)+w3(x1))/0.35)**2)
            + EPS)
    return energy + regularization_func(x)

def _normal_energy_func(x, mu=0., logvar=0.):
    energy = logvar + (x - mu)**2 / math.exp(logvar) + math.log(2.*math.pi)
    energy = 0.5 * energy
    return energy

def normal_energy_func(x, mu=0., logvar=0.):
    batch_size = x.size(0)
    x = x.view(batch_size, -1)
    return torch.sum(_normal_energy_func(x, mu, logvar), dim=1)

#def normal_energy_func(x,
#                       mu1=0., logvar1=0.,
#                       mu2=0., logvar2=0.,
#                       ):
#    assert x.dim() == 2
#    assert x.size(1) == 2
#    batch_size = x.size(0)
#    x = x.view(batch_size, -1)
#    x1 = x[:, :1]
#    x2 = x[:, 1:]
#
#    energy = _normal_energy_func(x1, mu1, logvar1) \
#            + _normal_energy_func(x2, mu2, logvar2)
#    return energy

def normal_prob(x, mu=0., std=1.):
    '''
    Inputs:⋅
        x: b1 x 1
        mu, logvar: scalar
    Outputs:
        prob: b1 x nz
    '''
    var = std**2
    logvar = math.log(var)
    logprob = - normal_energy_func(x, mu, logvar)
    #return logprob
    prob = torch.exp(logprob)
    return prob

#def energy_to_unnormalized_prob(energy):
#    prob = torch.exp(-energy) # unnormalized prob
#    return prob
#
#def get_data_for_heatmap(val=4, num=256):
#    _x = np.linspace(-val, val, num)
#    _y = np.linspace(-val, val, num)
#    _u, _v = np.meshgrid(_x, _y)
#    _data = np.stack([_u.reshape(num**2), _v.reshape(num**2)], axis=1)
#    return _data, _x, _y
#
#def run_energy_fun_for_vis(energy_func, num=256):
#    _z, _, _ = get_data_for_heatmap(num=num)
#    z = torch.from_numpy(_z)
#    prob = energy_func(z)
#    _prob = prob.cpu().float().numpy()
#    _prob = _prob.reshape(num, num)
#    return _prob
#
#def get_energy_func_plot(prob):
#    # plot
#    fig, ax = plt.subplots(figsize=(5, 5))
#    im = ax.imshow(prob, cmap='jet')
#    ax.grid(False)
#
#    # draw to canvas
#    fig.canvas.draw()  # draw the canvas, cache the renderer
#    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
#    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#
#    # close figure
#    plt.close()
#    return image

''' test '''
'''
import numpy as np
import torchvision.utils as vutils
from visualization import convert_npimage_torchimage, get_prob_from_energy_func_for_vis, get_imshow_plot

#energy_func = energy_func4
energy_func = normal_energy_func
_prob = get_prob_from_energy_func_for_vis(energy_func, num=256)
_img = get_imshow_plot(_prob)
filename='hi.png'
vutils.save_image(convert_npimage_torchimage(_img), filename)
'''
