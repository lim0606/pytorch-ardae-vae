import os
import sys
import argparse
import datetime
import time
import math
import glob

import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchcontrib

import torchvision.utils as vutils

import datasets as dset
import models as net
from utils import Adam
from utils import StepLR
from models.aux import aux_loss_for_grad

from utils import expand_tensor
from utils import load_checkpoint, save_checkpoint, load_end_iter, logging, get_time, annealing_func, EndIterError
from utils import convert_npimage_torchimage, get_scatter_plot, get_quiver_plot, get_data_for_quiver_plot, get_prob_from_energy_func_for_vis, get_imshow_plot, get_2d_histogram_plot, get_grid_image

from tensorboardX import SummaryWriter


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='swissroll',
                    choices=['swissroll', '25gaussians', 'sbmnist', 'dbmnist', 'dbmnist-val5k'],
                    help='dataset: swissroll | 25gaussians | sbmnist | dbmnist | dbmnist-val5k ')

# net architecture
parser.add_argument('--model', default='mlp-concat',
                    choices=['mlp-concat',
                             'mnist-concat',
                             'mnist-conv',
                             'resconv', 'resconvct', 'resconv-res', 'resconvct-res', 'resconv-res2', 'resconvct-res2', 'resconvct-res3', 'resconvct-res4',
                             'auxmlp',
                             'auxmnist',
                             'auxconv',
                             'auxresconv', 'auxresconvct', 'auxresconv-clip', 'auxresconvct-clip',
                             ],
                    help='model: mlp-concat '
                              '| mnist-concat '
                              '| mnist-conv '
                              '| resconv | resconvct | resconv-res | resconvct-res '
                              '| auxmlp '
                              '| auxmnist '
                              '| auxconv '
                              '| auxresconv | auxresconvct | auxresconv-clip | auxresconvct-clip '
                              )
parser.add_argument('--model-z-dim', type=int, default=2,
                    help='latent variable dim of encoder.')
parser.add_argument('--model-h-dim', type=int, default=128,
                    help='hidden dim of enc/dec networks.')
parser.add_argument('--model-n-dim', type=int, default=2,
                    help='noise source dim of encoder.')
parser.add_argument('--model-n-layers', type=int, default=2,
                    help='number of hidden layers.')
parser.add_argument('--model-nonlin', default='relu',
                    help='activation function')
parser.add_argument('--model-clip-z0-logvar', default='none',
                    choices=['none'],
                    help='clip logvar of z0 in encoder (hierarchical encoder)')
parser.add_argument('--model-clip-z-logvar', default='none',
                    choices=['none'],
                    help='clip logvar of z in encoder (hierarchical encoder)')
parser.add_argument('--cdae', default='mlp',
                    choices=['mlp', 'mlp-res', 'mlp-grad'],
                    help='cdae: mlp | mlp-res | mlp-grad')
parser.add_argument('--cdae-h-dim', type=int, default=128,
                    help='hidden dim of denoising autoencoder network.')
parser.add_argument('--cdae-n-layers', type=int, default=2,
                    help='number of hidden layers.')
parser.add_argument('--cdae-nonlin', default='relu',
                    help='activation function')
parser.add_argument('--cdae-ctx-type', default='data',
                    choices=['data', 'lt0', 'hidden1a',],
                    help='flag context type for cdae')

# conditional dae
parser.add_argument('--std-scale', type=float, default=1.0,
                    help='std scaling for denoising autoencoder')
parser.add_argument('--delta', type=float, default=1,
                    help='prior variance for std sampling distribution')
parser.add_argument('--num-cdae-updates', type=int, default=1,
                    help='number of cdae updates')

# type of data
parser.add_argument('--nheight', type=int, default=1,
                    help='the height / width of the input to network')
parser.add_argument('--nchannels', type=int, default=2,
                    help='number of channels in input')

# training
parser.add_argument('--m-lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--d-lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--d-lr-min', type=float, default=0.0001,
                    help='min learning rate')
parser.add_argument('--epochs', type=int, default=30,
                    help='upper epoch limit')
parser.add_argument('--train-batch-size', type=int, default=1024, metavar='N',
                    help='input batch size for training (default: 20)')
parser.add_argument('--eval-batch-size', type=int, default=1024, metavar='N',
                    help='input batch size for test (default: 10)')
parser.add_argument('--start-epoch', type=int, default=1,
                    help='start epoch')
parser.add_argument('--start-batch-idx', type=int, default=0,
                    help='start batch-idx')
parser.add_argument('--train-nz-cdae', type=int, default=1, metavar='N',
                    help='the number of z samples per data point (default: 1)')
parser.add_argument('--train-nz-model', type=int, default=1, metavar='N',
                    help='the number of z samples per data point (default: 1)')
parser.add_argument('--train-nstd-cdae', type=int, default=1, metavar='N',
                    help='the number of z samples per data point (default: 1)')

parser.add_argument('--m-optimizer', default='adam',
                    choices=['sgd', 'adam', 'amsgrad', 'rmsprop'],
                    help='optimization methods: sgd | adam | amsgrad | rmsprop ')
parser.add_argument('--m-beta1', type=float, default=0.5, help='beta1 for adam or adam-amsgrad. default=0.5') # sgd or rmsprop
parser.add_argument('--m-momentum', type=float, default=0.5, help='momentum for std or rmsprop. default=0.5') # adam
parser.add_argument('--d-optimizer', default='adam',
                    choices=['sgd', 'adam', 'amsgrad', 'rmsprop'],
                    help='optimization methods: sgd | adam | amsgrad | rmsprop ')
parser.add_argument('--d-beta1', type=float, default=0.5, help='beta1 for adam or adam-amsgrad. default=0.5') # sgd or rmsprop
parser.add_argument('--d-momentum', type=float, default=0.5, help='momentum for std or rmsprop. default=0.5') # adam

# training (beta, eta, and lmbd annealing)
parser.add_argument('--beta-init', type=float, default=1.0,
                    help='initial beta value for beta annealing')
parser.add_argument('--beta-fin', type=float, default=1.0,
                    help='final beta value for beta annealing')
parser.add_argument('--beta-annealing', type=float, default=None,
                    help='interval to annealing beta')
parser.add_argument('--eta-init', type=float, default=0.,
                    help='initial eta value for eta annealing')
parser.add_argument('--eta-fin', type=float, default=0.,
                    help='final eta value for eta annealing')
parser.add_argument('--eta-annealing', type=float, default=None,
                    help='interval to annealing eta')
parser.add_argument('--lmbd-init', type=float, default=0.,
                    help='initial lmbd value for lmbd annealing')
parser.add_argument('--lmbd-fin', type=float, default=0.,
                    help='final lmbd value for lmbd annealing')
parser.add_argument('--lmbd-annealing', type=float, default=None,
                    help='interval to annealing lmbd')

# eval
parser.add_argument('--iws-samples', type=int, default=512,
                    help='number of iwae samples (default: 512)')
parser.add_argument('--m-weight-avg', default='none',
                    choices=['none', 'swa', 'polyak'],
                    help='weight average method (evaluate): swa | polyak')
parser.add_argument('--m-weight-avg-start', type=int, default=1000,
                    help='weight average method (evaluate): swa | polyak')
parser.add_argument('--m-weight-avg-decay', type=float, default=0.998,
                    help='weight average method (evaluate): swa | polyak')

# final mode
parser.add_argument('--train-mode', default='train',
                    choices=['train', 'final'],
                    help='training mode: train | final')

# log
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=100,
                    help='log print-out interval (iter)')
parser.add_argument('--vis-interval', type=int, default=1000,
                    help='visualization interval (iter)')
parser.add_argument('--eval-iws-interval', type=int, default=1000,
                    help='evaluation interval (iter)')
parser.add_argument('--ckpt-interval', type=int, default=10000,
                    help='checkpoint interval (iter)')
parser.add_argument('--sav-interval', type=int, default=0,
                    help='model save interval (epoch)')

# save
parser.add_argument('--resume', dest='resume', action='store_true', default=True,
                    help='flag to resume the experiments')
parser.add_argument('--no-resume', dest='resume', action='store_false', default=True,
                    help='flag to resume the experiments')
parser.add_argument('--cache', default=None, help='path to cache')
parser.add_argument('--experiment', default=None, help='name of experiment')
parser.add_argument('--exp-num', type=int, default=None,
                    help='experiment number')

# parse arguments
opt = parser.parse_args()

# preprocess arguments
opt.cuda = not opt.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if opt.cuda else "cpu")
opt.best_val_loss = None
if opt.beta_annealing is None or opt.beta_annealing < 1:
    opt.beta_annealing = None
if opt.eta_annealing is None or opt.eta_annealing < 1:
    opt.eta_annealing = None
if opt.lmbd_annealing is None or opt.lmbd_annealing < 1:
    opt.lmbd_annealing = None

# generate cache folder
if opt.cache is None:
    opt.cache = 'experiments'
if opt.experiment is None:
    opt.experiment = '-'.join(['m{}-mz{}-mh{}-mn{}-mnh{}-ma{}'.format(
                                   opt.model,
                                   opt.model_z_dim,
                                   opt.model_h_dim,
                                   opt.model_n_dim,
                                   opt.model_n_layers,
                                   'sfp' if opt.model_nonlin == 'softplus' else opt.model_nonlin,
                                   #'-mz0cl{}-mzcl{}'.format(
                                   #    opt.model_clip_z0_logvar,
                                   #    opt.model_clip_z_logvar,
                                   #)
                                   ),
                               'd{}-dh{}-dnh{}-da{}-dct{}'.format(
                                   opt.cdae,
                                   opt.cdae_h_dim,
                                   opt.cdae_n_layers,
                                   'sfp' if opt.cdae_nonlin == 'softplus' else opt.cdae_nonlin,
                                   opt.cdae_ctx_type,
                                   ),
                               'm{}-bt1{}'.format(opt.m_optimizer, opt.m_beta1) if opt.m_optimizer in ['adam', 'amsgrad'] else 'm{}-mt{}'.format(opt.m_optimizer, opt.m_momentum),
                               'mlr{}'.format(opt.m_lr),
                               'd{}-bt1{}'.format(opt.d_optimizer, opt.d_beta1) if opt.d_optimizer in ['adam', 'amsgrad'] else 'd{}-mt{}'.format(opt.d_optimizer, opt.d_momentum),
                               'dlr{}'.format(opt.d_lr),
                               'tbs{}'.format(opt.train_batch_size),
                               'nd{}'.format(opt.num_cdae_updates),
                               'mwa{}{}'.format(opt.m_weight_avg,
                                               '-was{}-wad{}'.format(opt.m_weight_avg_start, opt.m_weight_avg_decay) if opt.m_weight_avg != 'none' else '',
                                               ),
                               'binit{}-bfin{}-bann{:d}'.format(
                                   opt.beta_init if opt.beta_init != opt.beta_fin else 1.,
                                   opt.beta_fin,
                                   int(opt.beta_annealing) if opt.beta_annealing is not None and opt.beta_init != opt.beta_fin else 0,
                                   ),
                               #'etinit{}-etfin{}-etann{:d}'.format(
                               #    opt.eta_init,
                               #    opt.eta_fin,
                               #    int(opt.eta_annealing) if opt.eta_annealing is not None else 0),
                               #'ldinit{}-ldfin{}-ldann{:d}'.format(
                               #    opt.lmbd_init,
                               #    opt.lmbd_fin,
                               #    int(opt.lmbd_annealing) if opt.lmbd_annealing is not None else 0),
                               'ssc{}'.format(opt.std_scale),
                               'del{}'.format(opt.delta),
                               'nzc{}{}'.format(
                                       opt.train_nz_cdae,
                                       '-nzs{}'.format(opt.train_nstd_cdae) if opt.train_nstd_cdae > 1 else '',
                                       ),
                               'nzm{}'.format(opt.train_nz_model),
                               '{}'.format(opt.exp_num if opt.exp_num else 0), #'exp{}'.format(opt.exp_num if opt.exp_num else 0),
                               ])
opt.path = os.path.join(opt.cache, opt.experiment)
if opt.resume:
    listing = glob.glob(opt.path+'-19*') + glob.glob(opt.path+'-20*')
    if len(listing) == 0:
        opt.path = '{}-{}'.format(opt.path, get_time())
    else:
        path_sorted = sorted(listing, key=lambda x: datetime.datetime.strptime(x, opt.path+'-%y%m%d-%H:%M:%S'))
        opt.path = path_sorted[-1]
        pass
else:
    opt.path = '{}-{}'.format(opt.path, get_time())
os.system('mkdir -p {}'.format(opt.path))

# print args
logging(str(opt), path=opt.path)

# init tensorboard
writer = SummaryWriter(opt.path)


# final mode
if opt.train_mode == 'final':
    opt.end_iter = load_end_iter(opt, filename='best-model-checkpoint.pth.tar', device=device)
else:
    opt.end_iter = None


# init dataset
train_loader, val_loader, test_loader, _ = dset.get_dataset(opt.dataset, opt.train_batch_size, opt.eval_batch_size, opt.cuda, final_mode=(opt.train_mode=='final'))


# init model
if opt.model == 'mlp-concat':
    model = net.ToyIPVAE(
            input_dim=opt.nchannels*opt.nheight*opt.nheight,
            noise_dim=opt.model_n_dim,
            h_dim=opt.model_h_dim,
            num_hidden_layers=opt.model_n_layers,
            nonlinearity=opt.model_nonlin,
            enc_type='concat',
            z_dim=opt.model_z_dim,
            ).to(device)
elif opt.model == 'mnist-concat':
    model = net.MNISTIPVAE(
            input_dim=opt.nchannels*opt.nheight*opt.nheight,
            noise_dim=opt.model_n_dim,
            h_dim=opt.model_h_dim,
            num_hidden_layers=opt.model_n_layers,
            nonlinearity=opt.model_nonlin,
            enc_type='concat',
            z_dim=opt.model_z_dim,
            ).to(device)
elif opt.model == 'mnist-conv':
    model = net.ConvIPVAE(
            input_height=opt.nheight,
            input_channels=opt.nchannels,
            z_dim=opt.model_z_dim,
            noise_dim=opt.model_n_dim,
            nonlinearity=opt.model_nonlin,
            ).to(device)
elif opt.model == 'resconv':
    model = net.ResConvIPVAE(
            input_height=opt.nheight,
            input_channels=opt.nchannels,
            z_dim=opt.model_z_dim,
            h_dim=opt.model_h_dim,
            num_hidden_layers=opt.model_n_layers,
            noise_dim=opt.model_n_dim,
            nonlinearity=opt.model_nonlin,
            do_center=False,
            enc_type='mlp',
            ).to(device)
elif opt.model == 'resconvct':
    model = net.ResConvIPVAE(
            input_height=opt.nheight,
            input_channels=opt.nchannels,
            z_dim=opt.model_z_dim,
            h_dim=opt.model_h_dim,
            num_hidden_layers=opt.model_n_layers,
            noise_dim=opt.model_n_dim,
            nonlinearity=opt.model_nonlin,
            do_center=True,
            enc_type='mlp',
            ).to(device)
elif opt.model == 'resconv-res':
    model = net.ResConvIPVAE(
            input_height=opt.nheight,
            input_channels=opt.nchannels,
            z_dim=opt.model_z_dim,
            h_dim=opt.model_h_dim,
            num_hidden_layers=opt.model_n_layers,
            noise_dim=opt.model_n_dim,
            nonlinearity=opt.model_nonlin,
            do_center=False,
            enc_type='res-wn-mlp',
            ).to(device)
elif opt.model == 'resconvct-res':
    model = net.ResConvIPVAE(
            input_height=opt.nheight,
            input_channels=opt.nchannels,
            z_dim=opt.model_z_dim,
            h_dim=opt.model_h_dim,
            num_hidden_layers=opt.model_n_layers,
            noise_dim=opt.model_n_dim,
            nonlinearity=opt.model_nonlin,
            do_center=True,
            enc_type='res-wn-mlp',
            ).to(device)
elif opt.model == 'resconv-res2':
    model = net.ResConvIPVAE(
            input_height=opt.nheight,
            input_channels=opt.nchannels,
            z_dim=opt.model_z_dim,
            h_dim=opt.model_h_dim,
            num_hidden_layers=opt.model_n_layers,
            noise_dim=opt.model_n_dim,
            nonlinearity=opt.model_nonlin,
            do_center=False,
            enc_type='res-mlp',
            ).to(device)
elif opt.model == 'resconvct-res2':
    model = net.ResConvIPVAE(
            input_height=opt.nheight,
            input_channels=opt.nchannels,
            z_dim=opt.model_z_dim,
            h_dim=opt.model_h_dim,
            num_hidden_layers=opt.model_n_layers,
            noise_dim=opt.model_n_dim,
            nonlinearity=opt.model_nonlin,
            do_center=True,
            enc_type='res-mlp',
            ).to(device)
elif opt.model == 'resconv-res3':
    model = net.ResConvIPVAE(
            input_height=opt.nheight,
            input_channels=opt.nchannels,
            z_dim=opt.model_z_dim,
            h_dim=opt.model_h_dim,
            num_hidden_layers=opt.model_n_layers,
            noise_dim=opt.model_n_dim,
            nonlinearity=opt.model_nonlin,
            do_center=False,
            enc_type='res-wn-mlp-lin',
            ).to(device)
elif opt.model == 'resconvct-res3':
    model = net.ResConvIPVAE(
            input_height=opt.nheight,
            input_channels=opt.nchannels,
            z_dim=opt.model_z_dim,
            h_dim=opt.model_h_dim,
            num_hidden_layers=opt.model_n_layers,
            noise_dim=opt.model_n_dim,
            nonlinearity=opt.model_nonlin,
            do_center=True,
            enc_type='res-wn-mlp-lin',
            ).to(device)
elif opt.model == 'resconv-res4':
    model = net.ResConvIPVAE(
            input_height=opt.nheight,
            input_channels=opt.nchannels,
            z_dim=opt.model_z_dim,
            h_dim=opt.model_h_dim,
            num_hidden_layers=opt.model_n_layers,
            noise_dim=opt.model_n_dim,
            nonlinearity=opt.model_nonlin,
            do_center=False,
            enc_type='res-mlp-lin',
            ).to(device)
elif opt.model == 'resconvct-res4':
    model = net.ResConvIPVAE(
            input_height=opt.nheight,
            input_channels=opt.nchannels,
            z_dim=opt.model_z_dim,
            h_dim=opt.model_h_dim,
            num_hidden_layers=opt.model_n_layers,
            noise_dim=opt.model_n_dim,
            nonlinearity=opt.model_nonlin,
            do_center=True,
            enc_type='res-mlp-lin',
            ).to(device)
elif opt.model == 'auxmlp':
    model = net.ToyAuxIPVAE(
            input_dim=opt.nchannels*opt.nheight*opt.nheight,
            noise_dim=opt.model_n_dim,
            h_dim=opt.model_h_dim,
            num_hidden_layers=opt.model_n_layers,
            nonlinearity=opt.model_nonlin,
            enc_type='simple',
            z_dim=opt.model_z_dim,
            clip_z0_logvar=opt.model_clip_z0_logvar,
            clip_z_logvar=opt.model_clip_z_logvar,
            ).to(device)
elif opt.model == 'auxmnist':
    model = net.MNISTAuxIPVAE(
            input_dim=opt.nchannels*opt.nheight*opt.nheight,
            noise_dim=opt.model_n_dim,
            h_dim=opt.model_h_dim,
            num_hidden_layers=opt.model_n_layers,
            nonlinearity=opt.model_nonlin,
            enc_type='simple',
            z_dim=opt.model_z_dim,
            clip_z0_logvar=opt.model_clip_z0_logvar,
            clip_z_logvar=opt.model_clip_z_logvar,
            ).to(device)
elif opt.model == 'auxconv':
    assert opt.model_h_dim == 0
    assert opt.model_n_layers == 0
    assert opt.model_clip_z0_logvar == 'none'
    assert opt.model_clip_z_logvar == 'none'
    model = net.MNISTConvAuxIPVAE(
            input_height=opt.nheight,
            input_channels=opt.nchannels,
            z0_dim=opt.model_n_dim,
            z_dim=opt.model_z_dim,
            nonlinearity=opt.model_nonlin,
            ).to(device)
elif opt.model == 'auxresconv':
    assert opt.model_h_dim == 0
    assert opt.model_n_layers == 0
    assert opt.model_clip_z0_logvar == 'none'
    assert opt.model_clip_z_logvar == 'none'
    model = net.MNISTResConvAuxIPVAE(
            input_height=opt.nheight,
            input_channels=opt.nchannels,
            z_dim=opt.model_z_dim,
            c_dim=450,
            z0_dim=opt.model_n_dim,
            nonlinearity=opt.model_nonlin,
            do_center=False,
            ).to(device)
elif opt.model == 'auxresconvct':
    assert opt.model_h_dim == 0
    assert opt.model_n_layers == 0
    assert opt.model_clip_z0_logvar == 'none'
    assert opt.model_clip_z_logvar == 'none'
    model = net.MNISTResConvAuxIPVAE(
            input_height=opt.nheight,
            input_channels=opt.nchannels,
            z_dim=opt.model_z_dim,
            c_dim=450,
            z0_dim=opt.model_n_dim,
            nonlinearity=opt.model_nonlin,
            do_center=True,
            ).to(device)
elif opt.model == 'auxresconv-clip':
    assert opt.model_h_dim == 0
    assert opt.model_n_layers == 0
    assert opt.model_clip_z0_logvar == 'none'
    assert opt.model_clip_z_logvar == 'none'
    model = net.MNISTResConvAuxIPVAEClipped(
            input_height=opt.nheight,
            input_channels=opt.nchannels,
            z_dim=opt.model_z_dim,
            c_dim=450,
            z0_dim=opt.model_n_dim,
            nonlinearity=opt.model_nonlin,
            do_center=False,
            ).to(device)
elif opt.model == 'auxresconvct-clip':
    assert opt.model_h_dim == 0
    assert opt.model_n_layers == 0
    assert opt.model_clip_z0_logvar == 'none'
    assert opt.model_clip_z_logvar == 'none'
    model = net.MNISTResConvAuxIPVAEClipped(
            input_height=opt.nheight,
            input_channels=opt.nchannels,
            z_dim=opt.model_z_dim,
            c_dim=450,
            z0_dim=opt.model_n_dim,
            nonlinearity=opt.model_nonlin,
            do_center=True,
            ).to(device)
else:
    raise NotImplementedError
logging(str(model), path=opt.path)

''' temporary '''
_prob = get_prob_from_energy_func_for_vis(model.energy_func, num=256)
_gtlatent = get_imshow_plot(_prob, val=6 if opt.dataset in ['mnist', 'sbmnist', 'dbmnist', 'dbmnist-val5k'] else 4, use_grid=False)
#img = convert_npimage_torchimage(_img)
#writer.add_image('train/latent', img.float(), 0)
''' --------- '''

# init optimizer
if opt.m_optimizer == 'sgd':
    model_optimizer = optim.SGD(model.parameters(), lr=opt.m_lr)
elif opt.m_optimizer == 'adam':
    model_optimizer = Adam(model.parameters(), lr=opt.m_lr, betas=(opt.m_beta1, 0.999))
elif opt.m_optimizer == 'amsgrad':
    model_optimizer = Adam(model.parameters(), lr=opt.m_lr, betas=(opt.m_beta1, 0.999), amsgrad=True)
elif opt.m_optimizer == 'rmsprop':
    model_optimizer = optim.RMSprop(model.parameters(), lr=opt.m_lr, momentum=opt.d_momentum)
else:
    raise NotImplementedError('unknown optimizer: {}'.format(opt.model_optimizer))
model_scheduler = None

# init weight avg
if opt.m_weight_avg == 'polyak':
    model_optimizer = torchcontrib.optim.Polyak(model_optimizer, polyak_start=opt.m_weight_avg_start, polyak_freq=1, polyak_decay=opt.m_weight_avg_decay)
elif opt.m_weight_avg == 'swa':
    model_optimizer = torchcontrib.optim.SWA(model_optimizer, swa_start=opt.m_weight_avg_start, swa_freq=1)
else:
    pass

# init cdae
if opt.cdae_ctx_type == 'data':
    context_dim = opt.nchannels*opt.nheight*opt.nheight
elif opt.cdae_ctx_type in ['lt0']:
    context_dim = opt.model_z_dim
elif opt.cdae_ctx_type in ['hidden1a']:
    if opt.model in ['auxmlp', 'auxmnist']:
        context_dim = opt.model_h_dim*2
    elif opt.model in ['auxconv']:
        context_dim = 800*2
    elif opt.model in ['auxresconv', 'auxresconvct', 'auxresconv-clip', 'auxresconvct-clip']:
        context_dim = 450#*2
    else:
        context_dim = opt.model_h_dim
else:
    raise NotImplementedError
if opt.cdae == 'mlp-res':
    cdae = net.MLPResCARDAE(
            input_dim=opt.model_z_dim,
            context_dim=context_dim,
            std=1.,
            h_dim=opt.cdae_h_dim,
            num_hidden_layers=opt.cdae_n_layers,
            nonlinearity=opt.cdae_nonlin,
            noise_type='gaussian',
            enc_ctx=True,
            enc_input=True,
            ).to(device)
elif opt.cdae == 'mlp-grad':
    cdae = net.MLPGradCARDAE(
            input_dim=opt.model_z_dim,
            context_dim=context_dim,
            std=1.,
            h_dim=opt.cdae_h_dim,
            num_hidden_layers=opt.cdae_n_layers,
            nonlinearity=opt.cdae_nonlin,
            noise_type='gaussian',
            enc_ctx=True,
            enc_input=True,
            ).to(device)
else:
    raise NotImplementedError
logging(str(cdae), path=opt.path)

# init params
cdae_params = list(cdae.parameters())
if opt.cdae_ctx_type in ['data', 'lt0', 'hidden1a']:
    pass
else:
    raise NotImplementedError

# init optimizer
if opt.d_optimizer == 'sgd':
    cdae_optimizer = optim.SGD(cdae_params, lr=opt.d_lr)
elif opt.d_optimizer == 'adam':
    cdae_optimizer = Adam(cdae_params, lr=opt.d_lr, betas=(opt.d_beta1, 0.999))
elif opt.d_optimizer == 'amsgrad':
    cdae_optimizer = Adam(cdae_params, lr=opt.d_lr, betas=(opt.d_beta1, 0.999), amsgrad=True)
elif opt.d_optimizer == 'rmsprop':
    cdae_optimizer = optim.RMSprop(cdae_params, lr=opt.d_lr, momentum=opt.d_momentum)
else:
    raise NotImplementedError('unknown optimizer: {}'.format(opt.cdae_optimizer))
cdae_scheduler = None

# resume
load_checkpoint(
        model, optimizer=model_optimizer, scheduler=model_scheduler,
        opt=opt, device=device,
        filename='{}model-checkpoint.pth.tar'.format('final-' if opt.train_mode == 'final' else ''),
        )
load_checkpoint(
        cdae, optimizer=cdae_optimizer, scheduler=cdae_scheduler,
        opt=opt,device=device,
        filename='{}cdae-checkpoint.pth.tar'.format('final-' if opt.train_mode == 'final' else ''),
        )

# define evaluate
def evaluate_iws(eval_loader, model, model_optimizer, name='valid'):
    model.eval()
    if opt.m_weight_avg != 'none':
        model_optimizer.use_buf()
    total_loss = 0.
    total_elbo = 0.
    total_logprob = 0.
    num_data = 0
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (eval_data, _) in enumerate(eval_loader):
            # init
            batch_size = eval_data.size(0)

            # init data
            eval_data = eval_data.to(device)

            # logprob
            logprob = model.logprob(eval_data, sample_size=opt.iws_samples)

            # add to total loss
            total_logprob += logprob.item() * batch_size
            num_data += batch_size

    # return
    elapsed = time.time() - start_time
    model.train()
    if opt.m_weight_avg != 'none':
        model_optimizer.use_sgd()
    return total_logprob / num_data, elapsed

# define train
def train(train_loader,
          model, model_optimizer,
          cdae,  cdae_optimizer,
          epoch, start_batch_idx=0,
          ):
    # set global variable
    global running_train_data_iter

    # init
    total_model_loss = 0.
    total_recon_loss = 0.
    total_prior_loss = 0.
    num_data_model = 0
    total_cdae_loss = 0.
    num_data_cdae = 0
    start_time = time.time()
    train_num_iters_per_epoch = len(train_loader.dataset) // opt.train_batch_size
    for _batch_idx in range(train_num_iters_per_epoch):
        # init batch_idx and i_ep
        batch_idx = _batch_idx + start_batch_idx
        i_ep = (epoch-1)*train_num_iters_per_epoch + batch_idx

        # end
        if opt.train_mode == 'final' and (i_ep+1) > opt.end_iter:
            raise EndIterError('end of training (final)')

        # init weights
        eta = annealing_func(opt.eta_init, opt.eta_fin, opt.eta_annealing, i_ep)
        beta  = annealing_func(opt.beta_init,  opt.beta_fin,  opt.beta_annealing,  i_ep)
        lmbd = annealing_func(opt.lmbd_init, opt.lmbd_fin, opt.lmbd_annealing, i_ep)

        ''' update cdae '''
        # init model and cdae
        model.train()
        cdae.train()

        # update cdae
        for i in range(opt.num_cdae_updates):
            # init grad
            cdae_optimizer.zero_grad()

            # init batch
            try:
                _train_data, _ = running_train_data_iter.next()
            except:
                running_train_data_iter = iter(train_loader)
                _train_data, _ = running_train_data_iter.next()

            # init data
            _train_data = _train_data.to(device)
            _batch_size = _train_data.size(0)

            # get context
            if opt.cdae_ctx_type == 'data':
                context = _train_data.unsqueeze(1)
                if 'mnist' in opt.dataset:
                    context = 2*context-1
                    context = context.view(_batch_size, 1, -1)
            elif opt.cdae_ctx_type == 'lt0':
                std00latent = model.encode(_train_data, std=0).detach()
                context = std00latent
            elif opt.cdae_ctx_type == 'hidden1a':
                hidden = model.encode.forward_hidden(_train_data, std=0).detach()
                context = hidden.unsqueeze(1)
            else:
                raise NotImplementedError

            # expand context
            if opt.train_nz_cdae > 1 and 'mlp' in opt.cdae:
                context = context.view(context.size(0), context.size(1), -1)

            # forward
            latent_mean = model.encode(_train_data, std=0).detach() # bsz x 1 x dims
            latent = model.forward_hidden(_train_data, nz=opt.train_nz_cdae)
            latent = latent.detach()

            # input
            latent_sub_mean = opt.std_scale*(latent-latent_mean)#.detach()
            std_qz = torch.std(latent_sub_mean, dim=1, keepdim=True) # bsz x 1 x dims
            std = opt.delta*torch.mean(std_qz, dim=2, keepdim=True) # bsz x 1 x 1
            cur_mean_std = std.mean().item()
            cur_mean_std_max = std.max().item()
            cur_mean_std_min = std.min().item()

            # init std
            stdmat = std*torch.randn(_batch_size, opt.train_nz_cdae*opt.train_nstd_cdae, 1, device=device)

            # forward
            sz = list(latent_sub_mean.size())
            _latent_sub_mean = latent_sub_mean.unsqueeze(2).expand(
                _batch_size, opt.train_nz_cdae, opt.train_nstd_cdae, sz[-1],
                ).reshape(_batch_size, opt.train_nz_cdae*opt.train_nstd_cdae, sz[-1])
            output, cdae_loss = cdae(_latent_sub_mean, context, std=stdmat, scale=opt.std_scale)

            # backward
            cdae_loss.backward()

            # add to total loss
            cur_cdae_loss = cdae_loss.item()
            total_cdae_loss += cur_cdae_loss * _batch_size
            num_data_cdae += _batch_size

            # update
            cdae_optimizer.step()

        ''' update model '''
        # init model and cdae
        model.train()
        cdae.eval()

        # init grad
        model_optimizer.zero_grad()

        # init batch
        try:
            _train_data, _ = running_train_data_iter.next()
        except:
            running_train_data_iter = iter(train_loader)
            _train_data, _ = running_train_data_iter.next()

        # init data
        _train_data = _train_data.to(device)
        _batch_size = _train_data.size(0)

        # forward
        output, _, latent, model_loss, recon_loss, prior_loss = model(_train_data, beta=beta, eta=eta, lmbd=lmbd, nz=opt.train_nz_model)

        # backward
        model_loss.backward(retain_graph=True)

        # get context
        if opt.cdae_ctx_type == 'data':
            context = _train_data.unsqueeze(1)
            if 'mnist' in opt.dataset:
                context = 2*context-1
                context = context.view(_batch_size, 1, -1)
        elif opt.cdae_ctx_type == 'lt0':
            std00latent = model.encode(_train_data, std=0).detach()
            context = std00latent
        elif opt.cdae_ctx_type == 'hidden1a':
            hidden = model.encode.forward_hidden(_train_data, std=0).detach()
            context = hidden.unsqueeze(1)
        else:
            raise NotImplementedError

        # expand context
        if opt.train_nz_model > 1 and 'mlp' in opt.cdae:
            context = context.view(context.size(0), context.size(1), -1)

        # grad estimate
        latent_mean = model.encode(_train_data, std=0).detach() # bsz x 1 x dims
        latent_sub_mean = opt.std_scale*(latent-latent_mean).detach()
        stdmat = torch.zeros(_batch_size, opt.train_nz_model, 1, device=device).fill_(0)
        grad = cdae.glogprob(latent_sub_mean, context, std=stdmat, scale=opt.std_scale).detach()

        # aux_loss with cdae forward and backward
        #aux_loss = aux_loss_for_grad(opt.std_scale*(latent-latent_mean), beta*grad.detach()/float(_batch_size*opt.train_nz_model))
        #aux_loss.backward()
        (opt.std_scale*(latent-latent_mean)).backward(beta*grad.detach()/float(_batch_size*opt.train_nz_model))

        # add to total loss
        cur_model_loss = model_loss.item()
        total_model_loss += cur_model_loss * _batch_size
        num_data_model += _batch_size
        cur_recon_loss = recon_loss.item()
        cur_prior_loss = prior_loss.item()
        total_recon_loss += cur_recon_loss * _batch_size
        total_prior_loss += cur_prior_loss * _batch_size

        # update
        model_optimizer.step()

        ''' print '''
        #if (batch_idx+1) % opt.log_interval == 0:
        if (i_ep+1) % opt.log_interval == 0:
            # set log info
            elapsed = time.time() - start_time

            # get lr
            param_group = cdae_optimizer.param_groups[0]
            lr = param_group['lr']

            # print
            logging('| iter {:d} | epoch {:3d} | {:5d}/{:5d} | ms/step {:5.2f} '
                    '| dlr {:.5f} '
                    '| (eff) std {:5.3f} '
                    '| (true) std {:5.3f} '
                    '| (eff) max std {:5.3f} '
                    '| (eff) min std {:5.3f} '
                    '| beta {:5.3f} ' #'| (crs) beta {:5.3f} '
                    #'| eta {:5.3f} '
                    #'| lmbd {:5.3f} '
                    '| loss (vae) {:5.3f} '
                    '| loss (recon) {:5.3f} '
                    '| loss (prior) {:5.3f} '
                    '| loss (cdae) {:5.4f} '
                    .format(
                    i_ep+1,
                    epoch,
                    batch_idx+1, train_num_iters_per_epoch,
                    elapsed * 1000 / opt.log_interval,
                    lr,
                    cur_mean_std,
                    cur_mean_std / opt.std_scale,
                    cur_mean_std_max,
                    cur_mean_std_min,
                    beta,
                    #eta,
                    #lmbd,
                    cur_model_loss,# / _batch_size,
                    cur_recon_loss,# / _batch_size,
                    cur_prior_loss,# / _batch_size,
                    cur_cdae_loss,# / _batch_size,
                    ),
                    path=opt.path)

            # write to tensorboard
            writer.add_scalar('{}/model/loss/step'.format(opt.train_mode),  cur_model_loss, i_ep+1)#/ _batch_size, i_ep+1)
            writer.add_scalar('{}/model/recon/step'.format(opt.train_mode), cur_recon_loss, i_ep+1)#/ _batch_size, i_ep+1)
            writer.add_scalar('{}/model/prior/step'.format(opt.train_mode), cur_prior_loss, i_ep+1)#/ _batch_size, i_ep+1)
            writer.add_scalar('{}/model/beta/step'.format(opt.train_mode),  beta, i_ep+1)
            #writer.add_scalar('{}/model/eta/step'.format(opt.train_mode), eta, i_ep+1)
            #writer.add_scalar('{}/model/lmbd/step'.format(opt.train_mode), lmbd, i_ep+1)
            writer.add_scalar('{}/cdae/loss/step'.format(opt.train_mode), cur_cdae_loss, i_ep+1)
            writer.add_scalar('{}/cdae/std/eff/mean/step'.format(opt.train_mode), cur_mean_std, i_ep+1)
            writer.add_scalar('{}/cdae/std/true/mean/step'.format(opt.train_mode), cur_mean_std/opt.std_scale, i_ep+1)
            writer.add_scalar('{}/cdae/std/eff/max/step'.format(opt.train_mode), cur_mean_std_max, i_ep+1)
            writer.add_scalar('{}/cdae/std/true/max/step'.format(opt.train_mode), cur_mean_std_max/opt.std_scale, i_ep+1)
            writer.add_scalar('{}/cdae/std/eff/min/step'.format(opt.train_mode), cur_mean_std_min, i_ep+1)
            writer.add_scalar('{}/cdae/std/true/min/step'.format(opt.train_mode), cur_mean_std_min/opt.std_scale, i_ep+1)
            writer.add_scalar('{}/cdae/lr/step'.format(opt.train_mode), lr, i_ep+1)

            # reset log info
            start_time = time.time()

        ''' evaluate '''
        if opt.train_mode == 'train' and opt.eval_iws_interval > 0 and (i_ep+1) % opt.eval_iws_interval == 0:
            logprob, elapsed_evaluate = evaluate_iws(val_loader, model, model_optimizer, name='valid')
            writer.add_scalar('val/logprob/iws/step', logprob, i_ep+1)
            logging('-' * 89, path=opt.path)
            logging('| val       '
                    '| iter {:d} | epoch {:3d} | {:5d}/{:5d} | sec/step {:5.2f} '
                    '| logprob (iws) {:.4f} '
                    .format(
                    i_ep+1, epoch,
                    batch_idx+1, train_num_iters_per_epoch,
                    elapsed_evaluate,
                    logprob,
                    ),
                    path=opt.path)
            logging('-' * 89, path=opt.path)

            # Save the model if the validation loss is the best we've seen so far.
            if not opt.best_val_loss or logprob > opt.best_val_loss:
                opt.best_val_loss = logprob
                save_checkpoint({
                    'epoch': epoch+1 if (batch_idx+1) == train_num_iters_per_epoch else epoch,
                    'batch_idx': (batch_idx+1) % train_num_iters_per_epoch,
                    'train_num_iters_per_epoch': train_num_iters_per_epoch,
                    'model': opt.model,
                    'state_dict': model.state_dict(),
                    'best_val_loss': opt.best_val_loss,
                    'optimizer' : model_optimizer.state_dict(),
                    'scheduler' : model_scheduler.state_dict() if model_scheduler is not None else None,
                }, opt, is_best=False, filename='best-model-checkpoint.pth.tar')
                save_checkpoint({
                    'epoch': epoch+1 if (batch_idx+1) == train_num_iters_per_epoch else epoch,
                    'batch_idx': (batch_idx+1) % train_num_iters_per_epoch,
                    'train_num_iters_per_epoch': train_num_iters_per_epoch,
                    'cdae': opt.cdae,
                    'state_dict': cdae.state_dict(),
                    'best_val_loss': opt.best_val_loss,
                    'optimizer' : cdae_optimizer.state_dict(),
                    'scheduler' : cdae_scheduler.state_dict() if cdae_scheduler is not None else None,
                }, opt, is_best=False, filename='best-cdae-checkpoint.pth.tar')

        ''' visualize '''
        if (i_ep+1) % opt.vis_interval == 0:
            # check variance
            #_, _, latent, _, _, _ = model(_train_data, nz=64) # bsz x ssz x zdim
            latent = model.forward_hidden(_train_data, nz=64) # bsz x ssz x zdim
            logvar_qz = torch.log(torch.var(latent.detach(), dim=1) + 1e-10) # bsz x zdim
            _logvar_qz = logvar_qz.view(-1).cpu().numpy()
            _mean_logvar_qz = torch.mean(logvar_qz).item() 
            _med_logvar_qz  = torch.median(logvar_qz).item()
            writer.add_scalar('{}/enc/logvar_qz/mean/step'.format(opt.train_mode),   _mean_logvar_qz, i_ep+1)
            writer.add_scalar('{}/enc/logvar_qz/median/step'.format(opt.train_mode), _med_logvar_qz,  i_ep+1)
            __logvar_qz = logvar_qz.view(_batch_size, -1).cpu().numpy()
            writer.add_histogram('{}/enc/logvar_qz/hist/step'.format(opt.train_mode), _logvar_qz, i_ep+1)
            for ii in range(min(2,_batch_size)):
                writer.add_histogram('train{}/enc/logvar_qz/hist/step'.format(opt.train_mode).format(ii), __logvar_qz[ii], i_ep+1)

            # visualize
            if opt.dataset in ['swissroll', '25gaussians']:
                # data
                val = 6
                gens = []
                outputs = []
                latents = []
                std08latents = []
                std05latents = []
                std01latents = []
                std0latents = []
                datas = []
                for i in range(int(20000//opt.train_batch_size)+1):
                    try:
                        _train_data, _ = running_train_data_iter.next()
                    except:
                        running_train_data_iter = iter(train_loader)
                        _train_data, _ = running_train_data_iter.next()
                    _train_data = _train_data.to(device)
                    datas += [_train_data]

                    gen, _, _ = model.generate(opt.train_batch_size)
                    gens += [gen.detach()]

                    std0latent = model.encode(_train_data, std=0)
                    std0latents += [std0latent.detach()]
                    std01latent = model.encode(_train_data, std=0.1)
                    std01latents += [std01latent.detach()]
                    std05latent = model.encode(_train_data, std=0.5)
                    std05latents += [std05latent.detach()]
                    std08latent = model.encode(_train_data, std=0.8)
                    std08latents += [std08latent.detach()]

                    output, _, latent, _, _, _ = model(_train_data)#, eta=eta)
                    outputs += [output.detach()]
                    latents += [latent.detach()]
                gen = torch.cat(gens, dim=0).cpu()
                data = torch.cat(datas, dim=0).cpu()
                output = torch.cat(outputs, dim=0).cpu()
                latent = torch.cat(latents, dim=0).cpu().squeeze()
                std08latent = torch.cat(std08latents, dim=0).cpu().squeeze()
                std05latent = torch.cat(std05latents, dim=0).cpu().squeeze()
                std01latent = torch.cat(std01latents, dim=0).cpu().squeeze()
                std0latent = torch.cat(std0latents, dim=0).cpu().squeeze()

                _data   = get_scatter_plot(data.numpy(), xlim=val, ylim=val)
                _output = get_scatter_plot(output.numpy(), xlim=val, ylim=val)
                _gen    = get_scatter_plot(gen.numpy(), xlim=val, ylim=val)
                _img = np.concatenate((_data, _output, _gen), axis=1)
                img = convert_npimage_torchimage(_img)
                writer.add_image('{}/data-recon-gen/scatter'.format(opt.train_mode), img.float(), i_ep+1)
                _data   = get_2d_histogram_plot(data.numpy(), val=val, num=128, use_grid=False)
                _output = get_2d_histogram_plot(output.numpy(), val=val, num=128, use_grid=False)
                _gen    = get_2d_histogram_plot(gen.numpy(), val=val, num=128, use_grid=False)
                _img = np.concatenate((_data, _output, _gen), axis=1)
                img = convert_npimage_torchimage(_img)
                writer.add_image('{}/data-recon-gen/heatmap'.format(opt.train_mode), img.float(), i_ep+1)
                #writer.add_image('train/data/heatmap',  convert_npimage_torchimage(_data).float(), i_ep+1)
                #writer.add_image('train/recon/heatmap', convert_npimage_torchimage(_output).float(), i_ep+1)
                writer.add_image('train/gen/heatmap',   convert_npimage_torchimage(_gen).float(), i_ep+1)

                # latent
                val = 4
                _img = get_scatter_plot(latent.numpy(), xlim=val, ylim=val)
                img = convert_npimage_torchimage(_img)
                writer.add_image('{}/latent/scatter'.format(opt.train_mode), img.float(), i_ep+1)
                _latent = get_2d_histogram_plot(latent.numpy(), val=val, num=128, use_grid=False)
                _std08latent = get_2d_histogram_plot(std08latent.numpy(), val=val, num=128, use_grid=False)
                _std05latent = get_2d_histogram_plot(std05latent.numpy(), val=val, num=128, use_grid=False)
                _std01latent = get_2d_histogram_plot(std01latent.numpy(), val=val, num=128, use_grid=False)
                _std0latent = get_2d_histogram_plot(std0latent.numpy(), val=val, num=128, use_grid=False)
                _img = np.concatenate((_gtlatent, _latent), axis=1)
                img = convert_npimage_torchimage(_img)
                writer.add_image('{}/gt_latent/heatmap'.format(opt.train_mode), img.float(), i_ep+1)
                writer.add_image('{}/latent/heatmap'.format(opt.train_mode), convert_npimage_torchimage(_latent).float(), i_ep+1)
                _img = np.concatenate((_gtlatent, _latent, _std08latent, _std05latent, _std01latent, _std0latent), axis=1)
                img = convert_npimage_torchimage(_img)
                writer.add_image('{}/alllatent/heatmap'.format(opt.train_mode), img.float(), i_ep+1)

            elif opt.dataset in ['sbmnist', 'dbmnist', 'dbmnist-val5k']:
                # latent
                val = 6
                latents = []
                std08latents = []
                std05latents = []
                std01latents = []
                std0latents = []
                for i in range(int(20000//opt.train_batch_size)+1):
                    try:
                        _train_data, _ = running_train_data_iter.next()
                    except:
                        running_train_data_iter = iter(train_loader)
                        _train_data, _ = running_train_data_iter.next()
                    _train_data = _train_data.to(device)
                    _batch_size = _train_data.size(0)
                    std0latent = model.encode(_train_data, std=0)
                    std0latents += [std0latent.detach().view(_batch_size,-1)]
                    std01latent = model.encode(_train_data, std=0.1)
                    std01latents += [std01latent.detach().view(_batch_size,-1)]
                    std05latent = model.encode(_train_data, std=0.5)
                    std05latents += [std05latent.detach().view(_batch_size,-1)]
                    std08latent = model.encode(_train_data, std=0.8)
                    std08latents += [std08latent.detach().view(_batch_size,-1)]
                    output, omu, latent, _, _, _ = model(_train_data)#, eta=eta)
                    latents += [latent.detach().view(_batch_size,-1)]
                latent = torch.cat(latents, dim=0).cpu().squeeze()
                std08latent = torch.cat(std08latents, dim=0).cpu().squeeze()
                std05latent = torch.cat(std05latents, dim=0).cpu().squeeze()
                std01latent = torch.cat(std01latents, dim=0).cpu().squeeze()
                std0latent = torch.cat(std0latents, dim=0).cpu().squeeze()
                _img = get_scatter_plot(latent.numpy(), xlim=val, ylim=val)
                img = convert_npimage_torchimage(_img)
                writer.add_image('{}/latent/scatter'.format(opt.train_mode), img.float(), i_ep+1)
                _latent = get_2d_histogram_plot(latent.numpy(), val=val, num=128, use_grid=False)
                _std08latent = get_2d_histogram_plot(std08latent.numpy(), val=val, num=128, use_grid=False)
                _std05latent = get_2d_histogram_plot(std05latent.numpy(), val=val, num=128, use_grid=False)
                _std01latent = get_2d_histogram_plot(std01latent.numpy(), val=val, num=128, use_grid=False)
                _std0latent = get_2d_histogram_plot(std0latent.numpy(), val=val, num=128, use_grid=False)
                _img = np.concatenate((_gtlatent, _latent), axis=1)
                img = convert_npimage_torchimage(_img)
                writer.add_image('{}/latent/heatmap'.format(opt.train_mode), img.float(), i_ep+1)
                _img = np.concatenate((_gtlatent, _latent, _std08latent, _std05latent, _std01latent, _std0latent), axis=1)
                img = convert_npimage_torchimage(_img)
                writer.add_image('{}/alllatent/heatmap'.format(opt.train_mode), img.float(), i_ep+1)

                # data
                _train_data = _train_data.cpu().view(_batch_size, opt.nchannels, opt.nheight, opt.nheight)
                output = output.detach().cpu().view(_batch_size, opt.nchannels, opt.nheight, opt.nheight)
                omu = omu.detach().cpu().view(_batch_size, opt.nchannels, opt.nheight, opt.nheight)
                gen, gmu, _ = model.generate(opt.train_batch_size)
                gen = gen.detach().cpu().view(opt.train_batch_size, opt.nchannels, opt.nheight, opt.nheight)
                gmu = gmu.detach().cpu().view(opt.train_batch_size, opt.nchannels, opt.nheight, opt.nheight)
                img1a = get_grid_image(_train_data, _batch_size, opt.nchannels, opt.nheight)
                img1b = get_grid_image(output, _batch_size, opt.nchannels, opt.nheight)
                img1c = get_grid_image(gen, opt.train_batch_size, opt.nchannels, opt.nheight)
                img1 = torch.cat([img1a, img1b, img1c], dim=2)
                writer.add_image('{}/data-recon-gen/sample'.format(opt.train_mode), img1, i_ep+1)
                img2a = img1a
                img2b = get_grid_image(omu, _batch_size, opt.nchannels, opt.nheight)
                img2c = get_grid_image(gmu, opt.train_batch_size, opt.nchannels, opt.nheight)
                img2 = torch.cat([img2a, img2b, img2c], dim=2)
                writer.add_image('{}/data-recon-gen/mean'.format(opt.train_mode), img2, i_ep+1)
            else:
                pass

            # flush
            writer.flush()

        ''' save '''
        opt.ckpt_interval = train_num_iters_per_epoch if opt.ckpt_interval == None else opt.ckpt_interval
        if (i_ep+1) % opt.ckpt_interval == 0:
            # save model
            save_checkpoint({
                'epoch': epoch+1 if (batch_idx+1) == train_num_iters_per_epoch else epoch,
                'batch_idx': (batch_idx+1) % train_num_iters_per_epoch,
                'train_num_iters_per_epoch': train_num_iters_per_epoch,
                'model': opt.model,
                'state_dict': model.state_dict(),
                'best_val_loss': opt.best_val_loss,
                'optimizer' : model_optimizer.state_dict(),
                'scheduler' : model_scheduler.state_dict() if model_scheduler is not None else None,
            }, opt, is_best=False, filename='{}model-checkpoint.pth.tar'.format('final-' if opt.train_mode == 'final' else ''))
            save_checkpoint({
                'epoch': epoch+1 if (batch_idx+1) == train_num_iters_per_epoch else epoch,
                'batch_idx': (batch_idx+1) % train_num_iters_per_epoch,
                'train_num_iters_per_epoch': train_num_iters_per_epoch,
                'cdae': opt.cdae,
                'state_dict': cdae.state_dict(),
                'best_val_loss': opt.best_val_loss,
                'optimizer' : cdae_optimizer.state_dict(),
                'scheduler' : cdae_scheduler.state_dict() if cdae_scheduler is not None else None,
            }, opt, is_best=False, filename='{}cdae-checkpoint.pth.tar'.format('final-' if opt.train_mode == 'final' else ''))

        ''' end final mode '''
        if opt.train_mode == 'final' and (i_ep+1) % opt.end_iter == 0:
            # save model
            save_checkpoint({
                'epoch': epoch+1 if (batch_idx+1) == train_num_iters_per_epoch else epoch,
                'batch_idx': (batch_idx+1) % train_num_iters_per_epoch,
                'train_num_iters_per_epoch': train_num_iters_per_epoch,
                'model': opt.model,
                'state_dict': model.state_dict(),
                'best_val_loss': opt.best_val_loss,
                'optimizer' : model_optimizer.state_dict(),
                'scheduler' : model_scheduler.state_dict() if model_scheduler is not None else None,
            }, opt, is_best=False, filename='{}model-checkpoint.pth.tar'.format('final-' if opt.train_mode == 'final' else ''))
            save_checkpoint({
                'epoch': epoch+1 if (batch_idx+1) == train_num_iters_per_epoch else epoch,
                'batch_idx': (batch_idx+1) % train_num_iters_per_epoch,
                'train_num_iters_per_epoch': train_num_iters_per_epoch,
                'cdae': opt.cdae,
                'state_dict': cdae.state_dict(),
                'best_val_loss': opt.best_val_loss,
                'optimizer' : cdae_optimizer.state_dict(),
                'scheduler' : cdae_scheduler.state_dict() if cdae_scheduler is not None else None,
            }, opt, is_best=False, filename='{}cdae-checkpoint.pth.tar'.format('final-' if opt.train_mode == 'final' else ''))

            # flush
            writer.flush()
            raise EndIterError('end of training (final)')

        ''' end epoch '''
        if batch_idx+1 == train_num_iters_per_epoch:
            # write to tensorboard
            writer.add_scalar('{}/model/loss/epoch'.format(opt.train_mode),  total_model_loss / num_data_model, epoch)
            writer.add_scalar('{}/model/recon/epoch'.format(opt.train_mode), total_recon_loss / num_data_model, epoch)
            writer.add_scalar('{}/model/prior/epoch'.format(opt.train_mode), total_prior_loss / num_data_model, epoch)
            writer.add_scalar('{}/model/beta/epoch'.format(opt.train_mode),  beta, epoch)
            #writer.add_scalar('{}/model/eta/epoch'.format(opt.train_mode), eta, epoch)
            #writer.add_scalar('{}/model/lmbd/epoch'.format(opt.train_mode), lmbd, epoch)
            writer.add_scalar('{}/cdae/loss/epoch'.format(opt.train_mode), total_cdae_loss / num_data_cdae, epoch)

            # flush
            writer.flush()

            return total_model_loss / num_data_model, total_cdae_loss / num_data_cdae

''' main '''
# Loop over epochs
try:
    for epoch in range(opt.start_epoch, opt.epochs+1):
        epoch_start_time = time.time()

        # train
        train(train_loader,
              model, model_optimizer,
              cdae, cdae_optimizer,
              epoch, opt.start_batch_idx,
              )
        opt.start_batch_idx = 0

        # flush writer
        writer.flush()

except KeyboardInterrupt:
    # close writer
    writer.flush()

    # end
    logging('-' * 89, path=opt.path)
    logging('Exiting from training early', path=opt.path)
    logging('-' * 89, path=opt.path)

    # return
    sys.exit(0)

except EndIterError:
    # close writer
    writer.flush()

    # end
    logging('-' * 89, path=opt.path)
    logging('End of training (final)', path=opt.path)
    logging('-' * 89, path=opt.path)


''' test visualize '''
if opt.dataset in ['swissroll', '25gaussians']:
    logging('test visualize', path=opt.path)
    val = 6
    gens = []
    outputs = []
    latents = []
    std08latents = []
    std05latents = []
    std01latents = []
    std0latents = []
    datas = []
    for i in range(int(1000000//opt.train_batch_size)+1):
        try:
            _train_data, _ = running_train_data_iter.next()
        except:
            running_train_data_iter = iter(train_loader)
            _train_data, _ = running_train_data_iter.next()
        _train_data = _train_data.to(device)
        datas += [_train_data]

        gen, _, _ = model.generate(opt.train_batch_size)
        gens += [gen.detach()]

        std0latent = model.encode(_train_data, std=0)
        std0latents += [std0latent.detach()]
        std01latent = model.encode(_train_data, std=0.1)
        std01latents += [std01latent.detach()]
        std05latent = model.encode(_train_data, std=0.5)
        std05latents += [std05latent.detach()]
        std08latent = model.encode(_train_data, std=0.8)
        std08latents += [std08latent.detach()]

        output, _, latent, _, _, _ = model(_train_data)
        outputs += [output.detach()]
        latents += [latent.detach()]
    gen = torch.cat(gens, dim=0).cpu()
    data = torch.cat(datas, dim=0).cpu()
    output = torch.cat(outputs, dim=0).cpu()
    latent = torch.cat(latents, dim=0).cpu().squeeze()
    std08latent = torch.cat(std08latents, dim=0).cpu().squeeze()
    std05latent = torch.cat(std05latents, dim=0).cpu().squeeze()
    std01latent = torch.cat(std01latents, dim=0).cpu().squeeze()
    std0latent = torch.cat(std0latents, dim=0).cpu().squeeze()

    _data   = get_2d_histogram_plot(data.numpy(), val=val, num=256, use_grid=False)
    _output = get_2d_histogram_plot(output.numpy(), val=val, num=256, use_grid=False)
    _gen    = get_2d_histogram_plot(gen.numpy(), val=val, num=256, use_grid=False)
    _img = np.concatenate((_data, _output, _gen), axis=1)
    img = convert_npimage_torchimage(_img)
    writer.add_image('test/data-recon-gen/heatmap', img.float(), 0)
    writer.add_image('test/gen/heatmap', convert_npimage_torchimage(_gen).float(), 0)

    # latent
    val = 4
    _latent = get_2d_histogram_plot(latent.numpy(), val=val, num=256, use_grid=False)
    _std08latent = get_2d_histogram_plot(std08latent.numpy(), val=val, num=256, use_grid=False)
    _std05latent = get_2d_histogram_plot(std05latent.numpy(), val=val, num=256, use_grid=False)
    _std01latent = get_2d_histogram_plot(std01latent.numpy(), val=val, num=256, use_grid=False)
    _std0latent = get_2d_histogram_plot(std0latent.numpy(), val=val, num=256, use_grid=False)
    _img = np.concatenate((_gtlatent, _latent), axis=1)
    img = convert_npimage_torchimage(_img)
    writer.add_image('test/gt_latent/heatmap', img.float(), 0)
    writer.add_image('test/latent/heatmap', convert_npimage_torchimage(_latent).float(), 0)
    _img = np.concatenate((_gtlatent, _latent, _std08latent, _std05latent, _std01latent, _std0latent), axis=1)
    img = convert_npimage_torchimage(_img)
    writer.add_image('test/alllatent/heatmap', img.float(), 0)

    # close writer
    writer.close()

else:
    if opt.train_mode == 'final':
        load_checkpoint(
                model, optimizer=model_optimizer, scheduler=model_scheduler,
                opt=opt, filename='final-model-checkpoint.pth.tar', device=device,
                )
    else:
        load_checkpoint(
                model, optimizer=model_optimizer, scheduler=model_scheduler,
                opt=opt, filename='best-model-checkpoint.pth.tar', device=device,
                )
    logprob, elapsed_evaluate = evaluate_iws(test_loader, model, model_optimizer, name='test')
    writer.add_scalar('test/logprob/iws/step', logprob, 0)
    logging('-' * 89, path=opt.path)
    logging('| test       '
            '| sec/step {:5.2f} '
            '| logprob (iws) {:.4f} '
            .format(
            elapsed_evaluate,
            logprob,
            ),
            path=opt.path)
    logging('-' * 89, path=opt.path)

    # close writer
    writer.close()
