import numpy as np

import torch
import torchvision.utils as vutils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import pandas as pd
import seaborn as sns
sns.set()
sns.set_style('whitegrid')
sns.set_palette('colorblind')


def convert_npimage_torchimage(image):
    return 255*torch.transpose(torch.transpose(torch.from_numpy(image), 0, 2), 1, 2)

def get_scatter_plot(data, labels=None, n_classes=None, num_samples=1000, xlim=None, ylim=None):
    '''
    data   : 2d points, batch_size x data_dim (numpy array)
    labels : labels, batch_size (numpy array)
    '''
    batch_size, data_dim = data.shape
    num_samples = min(num_samples, batch_size)
    if labels is None:
        labels = np.zeros(batch_size, dtype=np.int)
    if n_classes is None:
        n_classes = len(np.unique(labels))

    # sub-samples
    if num_samples != batch_size:
        indices = np.random.permutation(batch_size)
        data = data[indices[:num_samples]]
        labels = labels[indices[:num_samples]]

    # init config
    palette = sns.color_palette(n_colors=n_classes)
    palette = [palette[i] for i in np.unique(labels)]

    # plot
    fig, ax = plt.subplots(figsize=(5, 5))
    data = {'x': data[:, 0],
            'y': data[:, 1],
            'class': labels}
    sns.scatterplot(x='x', y='y', hue='class', data=data, palette=palette)

    # set config
    if xlim is not None:
        plt.xlim((-xlim, xlim))
    if ylim is not None:
        plt.ylim((-ylim, ylim))

    # draw to canvas
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # close figure
    plt.close()
    return image

def get_data_for_quiver_plot(val, num):
    _x = np.linspace(-val, val, num)
    _y = np.linspace(-val, val, num)
    _u, _v = np.meshgrid(_x, _y)
    _vis_data = np.stack([_u.reshape(num**2), _v.reshape(num**2)], axis=1)
    vis_data = torch.from_numpy(_vis_data).float()
    return vis_data, _x, _y

def get_quiver_plot(vec, x_pos, y_pos, xlim=None, ylim=None, scale=None):
    '''
    vec   : 2d points, batch_size x data_dim (numpy array)
    pos   : 2d points, batch_size x data_dim (numpy array)
    '''
    grid_size = x_pos.shape[0]
    batch_size = vec.shape[0]
    assert batch_size == grid_size**2
    assert y_pos.shape[0] == grid_size

    # get x, y, u, v 
    X = x_pos #np.arange(-10, 10, 1)
    Y = y_pos #np.arange(-10, 10, 1)
    #U, V = np.meshgrid(X, Y)
    U = vec[:, 0].reshape(grid_size, grid_size)
    V = vec[:, 1].reshape(grid_size, grid_size)

    # plot
    fig, ax = plt.subplots(figsize=(5, 5))
    q = ax.quiver(X, Y, U, V, pivot='mid', scale=scale)
    #ax.quiverkey(q, X=0.3, Y=1.1, U=10,
    #                     label='Quiver key, length = 10', labelpos='E')

    # set config
    if xlim is not None:
        plt.xlim((-xlim, xlim))
    if ylim is not None:
        plt.ylim((-ylim, ylim))

    # tight
    plt.tight_layout()

    # draw to canvas
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # close figure
    plt.close()
    return image

def get_data_for_heatmap(val=4, num=256):
    _x = np.linspace(-val, val, num)
    _y = np.linspace(-val, val, num)
    _u, _v = np.meshgrid(_x, _y)
    _data = np.stack([_u.reshape(num**2), _v.reshape(num**2)], axis=1)
    return _data, _x, _y

def energy_to_unnormalized_prob(energy):
    prob = torch.exp(-energy) # unnormalized prob
    return prob

def get_prob_from_energy_func_for_vis(energy_func, val=4, num=256):
    # get grid
    _z, _, _ = get_data_for_heatmap(val=val, num=num)
    z = torch.from_numpy(_z).float()

    # run energy func
    energy = energy_func(z)
    prob = energy_to_unnormalized_prob(energy)

    # convert to numpy array
    _prob = prob.cpu().float().numpy()
    _prob = _prob.reshape(num, num)
    return _prob

def get_imshow_plot(prob, val=4, use_grid=True):
    # plot
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(prob, cmap='jet', extent=[-val, val, -val, val])
    ax.grid(False)
    if use_grid:
        plt.xticks(np.arange(-val, val+1, step=1))
        plt.yticks(np.arange(-val, val+1, step=1))
    else:
        plt.xticks([])
        plt.yticks([])

    # tight
    plt.tight_layout()

    # draw to canvas
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # close figure
    plt.close()
    return image

def get_1d_histogram_plot(data, val=4, num=256, use_grid=True):
    xmin = 0
    xmax = val

    # get data
    x = data

    # get histogram
    hist, xedges = np.histogram(x, range=[xmin, xmax], bins=num)

    # plot heatmap
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.bar(xedges[:-1], hist, width=0.5)#, color='#0504aa',alpha=0.7)

    ax.grid(False)
    if use_grid:
        plt.xticks(np.arange(0, val+1, step=1))
    else:
        plt.xticks([])

    # tight
    plt.tight_layout()

    # draw to canvas
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # close figure
    plt.close()
    return image

def get_2d_histogram_plot(data, val=4, num=256, use_grid=True):
    xmin = -val
    xmax = val
    ymin = -val
    ymax = val

    # get data
    x = data[:, 0]
    y = data[:, 1]

    # get histogram
    heatmap, xedges, yedges = np.histogram2d(x, y, range=[[xmin, xmax], [ymin, ymax]], bins=num)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # plot heatmap
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(heatmap.T, extent=extent, cmap='jet')
    ax.grid(False)
    if use_grid:
        plt.xticks(np.arange(-val, val+1, step=1))
        plt.yticks(np.arange(-val, val+1, step=1))
    else:
        plt.xticks([])
        plt.yticks([])

    # tight
    plt.tight_layout()

    # draw to canvas
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # close figure
    plt.close()
    return image

def get_grid_image(input, batch_size, nchannels, nheight, nwidth=None, ncol=8, pad_value=0, do_square=True):
    '''
    input : b x c x h x w (where h = w)
    '''
    if batch_size > ncol**2 and do_square:
        input = input[:ncol**2, :, :, :]
        batch_size = ncol**2
    nwidth = nwidth if nwidth is not None else nheight
    input = input.detach()
    output = input.view(batch_size, nchannels, nheight, nwidth).clone().cpu()
    output = vutils.make_grid(output, nrow=ncol, normalize=True, scale_each=True, pad_value=pad_value)
    #output = vutils.make_grid(output, normalize=False, scale_each=False)
    return output

#def get_canvas(fig):
#    fig.canvas.draw()  # draw the canvas, cache the renderer
#    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
#    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#    return image
#    # close figure
#    plt.close()
#
#def get_contour_with_batch_size(model, batch_size=128, vmin=-10.0, vmax=10.0, title=None):
#    model.eval()
#    matplotlib.rcParams['xtick.direction'] = 'out'
#    matplotlib.rcParams['ytick.direction'] = 'out'
#    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
#
#    # tmp
#    weight = next(model.parameters())
#
#    # gen grid⋅
#    delta = 0.1
#    xv, yv = torch.meshgrid([torch.arange(vmin, vmax, delta), torch.arange(vmin, vmax, delta)])
#    h = yv.size(0)
#    w = xv.size(0)
#    yv = yv.contiguous().view(-1)
#    xv = xv.contiguous().view(-1)
#    input = torch.cat([xv.unsqueeze(1), yv.unsqueeze(1)], dim=1).to(weight.device)
#
#    # forward
#    prob = model.prob(input, batch_size=batch_size)
#
#    # convert torch variable to numpy array
#    xv = xv.cpu().numpy().reshape(h, w)
#    yv = yv.cpu().numpy().reshape(h, w)
#    zv = prob.detach().cpu().numpy().reshape(h, w)
#
#    # plot and save⋅
#    fig = plt.figure()
#    CS1 = plt.contourf(xv, yv, zv)
#    CS2 = plt.contour(xv, yv, zv, alpha=.7, colors='k')
#    plt.clabel(CS2, inline=1, fontsize=10, colors='k')
#    #plt.title('Simplest default with labels')
#    if title is not None:
#        plt.title(title)
#    #plt.savefig(filename)
#    #plt.close()
#    image = get_canvas(fig)
#    plt.close()
#
#    return image
#
##def get_contour_with_data(model, data, vmin=-10.0, vmax=10.0, title=None):
##    model.eval()
##    matplotlib.rcParams['xtick.direction'] = 'out'
##    matplotlib.rcParams['ytick.direction'] = 'out'
##    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
##
##    # gen grid⋅
##    delta = 0.1
##    xv, yv = torch.meshgrid([torch.arange(vmin, vmax, delta), torch.arange(vmin, vmax, delta)])
##    h = yv.size(0)
##    w = xv.size(0)
##    yv = yv.contiguous().view(-1)
##    xv = xv.contiguous().view(-1)
##    input = torch.cat([xv.unsqueeze(1), yv.unsqueeze(1)], dim=1).to(data.device)
##
##    # forward
##    prob = model.prob(input, data)
##
##    # convert torch variable to numpy array
##    xv = xv.cpu().numpy().reshape(h, w)
##    yv = yv.cpu().numpy().reshape(h, w)
##    zv = prob.detach().cpu().numpy().reshape(h, w)
##
##    # plot and save⋅
##    fig = plt.figure()
##    CS1 = plt.contourf(xv, yv, zv)
##    CS2 = plt.contour(xv, yv, zv, alpha=.7, colors='k')
##    plt.clabel(CS2, inline=1, fontsize=10, colors='k')
##    #plt.title('Simplest default with labels')
##    if title is not None:
##        plt.title(title)
##    #plt.savefig(filename)
##    #plt.close()
##    image = get_canvas(fig)
##    plt.close()
##
##    return image
#
#def get_contour_with_z(model, z, vmin=-10.0, vmax=10.0, title=None):
#    model.eval()
#    matplotlib.rcParams['xtick.direction'] = 'out'
#    matplotlib.rcParams['ytick.direction'] = 'out'
#    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
#
#    # gen grid⋅
#    delta = 0.1
#    xv, yv = torch.meshgrid([torch.arange(vmin, vmax, delta), torch.arange(vmin, vmax, delta)])
#    h = yv.size(0)
#    w = xv.size(0)
#    yv = yv.contiguous().view(-1)
#    xv = xv.contiguous().view(-1)
#    input = torch.cat([xv.unsqueeze(1), yv.unsqueeze(1)], dim=1).to(z.device)
#
#    # forward
#    prob = model.prob(input, z=z)
#
#    # convert torch variable to numpy array
#    xv = xv.cpu().numpy().reshape(h, w)
#    yv = yv.cpu().numpy().reshape(h, w)
#    zv = prob.detach().cpu().numpy().reshape(h, w)
#
#    # plot and save⋅
#    fig = plt.figure()
#    CS1 = plt.contourf(xv, yv, zv)
#    CS2 = plt.contour(xv, yv, zv, alpha=.7, colors='k')
#    plt.clabel(CS2, inline=1, fontsize=10, colors='k')
#    #plt.title('Simplest default with labels')
#    if title is not None:
#        plt.title(title)
#    #plt.savefig(filename)
#    #plt.close()
#    image = get_canvas(fig)
#    plt.close()
#
#    return image
