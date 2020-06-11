'''
miscellaneous functions: learning
'''
import os
import datetime

import numpy as np

import torch
import torchvision.utils as vutils

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


''' expand tensor '''
def expand_tensor(input, sample_size, do_unsqueeze):
    batch_size = input.size(0)
    if do_unsqueeze:
        sz_from = [-1]*(input.dim()+1)
        sz_from[1] = sample_size
        input_expanded = input.unsqueeze(1).expand(*sz_from).contiguous()

        sz_to = list(input.size())
        sz_to[0] = batch_size*sample_size
    else:
        assert input.size(1) == 1
        sz_from = [-1]*(input.dim())
        sz_from[1] = sample_size
        input_expanded = input.expand(*sz_from).contiguous()

        _sz_to = list(input.size())
        sz_to = _sz_to[0:1]+_sz_to[2:]
        sz_to[0] = batch_size*sample_size
    input_expanded_flattend = input_expanded.view(*sz_to)
    return input_expanded, input_expanded_flattend

''' cont out size '''
def conv_out_size(hin, kernel_size, stride=1, padding=0, dilation=1):
    hout = (hin + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1
    return int(hout)

def deconv_out_size(hin, kernel_size, stride=1, padding=0, output_padding=0, dilation=1):
    hout = (hin-1)*stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1
    return int(hout)


''' annealing '''
def annealing_func(val_init, val_fin, val_annealing, step):
    val = val_init + (val_fin  - val_init)  / float(val_annealing)  * float(min(val_annealing, step)) if val_annealing is not None else val_fin
    return float(val)


''' for monitoring lr '''
def get_lrs(optimizer):
    lrs = [float(param_group['lr']) for param_group in optimizer.param_groups]
    lr_max = max(lrs)
    lr_min = min(lrs)
    return lr_min, lr_max


''' save and load '''
def save_checkpoint(state, opt, is_best, filename='checkpoint.pth.tar'):
    filename = os.path.join(opt.path, filename)
    print("=> save checkpoint '{}'".format(filename))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def load_checkpoint(model, optimizer, opt, filename='checkpoint.pth.tar', verbose=True, device=None, scheduler=None):
    filename = os.path.join(opt.path, filename)
    if os.path.isfile(filename):
        if verbose:
            print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location=device) if device is not None else torch.load(filename)
        opt.start_epoch = checkpoint['epoch']
        opt.start_batch_idx = checkpoint['batch_idx']
        opt.best_val_loss = checkpoint['best_val_loss']
        if 'train_num_iters_per_epoch' in checkpoint.keys():
            opt.train_num_iters_per_epoch = checkpoint['train_num_iters_per_epoch']
        if model is not None:
            model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if verbose:
            print("=> loaded checkpoint '{}'".format(filename))
        if 'start_std' in checkpoint.keys():
            opt.start_std = checkpoint['start_std']
    else:
        print("=> no checkpoint found at '{}'".format(filename))

def load_end_iter(opt, filename='best-checkpoint.pth.tar', verbose=True, device=None):
    filename = os.path.join(opt.path, filename)
    if os.path.isfile(filename):
        if verbose:
            print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location=device) if device is not None else torch.load(filename)
        start_epoch = checkpoint['epoch']
        start_batch_idx = checkpoint['batch_idx']
        train_num_iters_per_epoch = checkpoint['train_num_iters_per_epoch']
        i_ep = (start_epoch-1)*train_num_iters_per_epoch + start_batch_idx
        return i_ep-1
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(filename))

class EndIterError(Exception):
    pass


''' log '''
def logging(s, path=None, filename='log.txt'):
    # print
    print(s)

    # save
    if path is not None:
        assert path, 'path is not define. path: {}'.format(path)
        with open(os.path.join(path, filename), 'a+') as f_log:
            f_log.write(s + '\n')

def get_time():
    return datetime.datetime.now().strftime('%y%m%d-%H:%M:%S')
    #return datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f')
