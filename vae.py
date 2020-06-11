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

from utils import load_checkpoint, save_checkpoint, load_end_iter, logging, get_time, annealing_func, EndIterError
from utils import convert_npimage_torchimage, get_scatter_plot, get_quiver_plot, get_data_for_quiver_plot, get_prob_from_energy_func_for_vis, get_imshow_plot, get_2d_histogram_plot, get_grid_image

from tensorboardX import SummaryWriter


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='swissroll',
                    choices=['swissroll', '25gaussians', 'sbmnist', 'dbmnist', 'dbmnist-val5k'],
                    help='dataset: swissroll | 25gaussians | sbmnist | dbmnist | dbmnist-val5k ')

# net architecture
parser.add_argument('--model', default='mnist',
                    choices=['toy', 'toy-maf', 'mnist', 'conv', 'resconv', 'resconvct',
                             'auxtoy', 'auxmnist', 'auxconv', 'auxresconv', 'auxresconvct'],
                    help='model: toy | toy-maf | mnist | conv')
parser.add_argument('--model-z-dim', type=int, default=8,
                    help='latent variable dim of encoder.')
parser.add_argument('--model-h-dim', type=int, default=300,
                    help='hidden dim of enc/dec networks.')
parser.add_argument('--model-n-dim', type=int, default=0,
                    help='z0 dim of aux encoder.')
parser.add_argument('--model-n-layers', type=int, default=1,
                    help='number of hidden layers.')
parser.add_argument('--model-nonlin', default='softplus',
                    help='activation function')
parser.add_argument('--model-clip-logvar', default='none',
                    help='clip-logvar in encoder')

# type of data
parser.add_argument('--nheight', type=int, default=28,
                    help='the height / width of the input to network')
parser.add_argument('--nchannels', type=int, default=1,
                    help='number of channels in input')

# training
parser.add_argument('--lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=32000,
                    help='upper epoch limit')
parser.add_argument('--train-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 20)')
parser.add_argument('--eval-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for test (default: 10)')
parser.add_argument('--optimizer', default='adam',
                    choices=['sgd', 'adam', 'amsgrad', 'rmsprop'],
                    help='optimization methods: sgd | adam | amsgrad | rmsprop ')
parser.add_argument('--start-epoch', type=int, default=1,
                    help='start epoch')
parser.add_argument('--start-batch-idx', type=int, default=0,
                    help='start batch-idx')

# adam
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam or adam-amsgrad. default=0.5')
# sgd or rmsprop
parser.add_argument('--momentum', type=float, default=0.5, help='momentum for std or rmsprop. default=0.9')

# training
parser.add_argument('--beta-init', type=float, default=1.0, #0.0001,
                    help='scale multiplier for dae-based grad approx. initial beta value for beta annealing.')
parser.add_argument('--beta-fin', type=float, default=1.0,
                    help='scale multiplier for dae-based grad approx. final beta value for beta annealing')
parser.add_argument('--beta-annealing', type=float, default=None, #50000, #None,
                    help='scale multiplier for dae-based grad approx. interval to annealing beta')

# eval
parser.add_argument('--iws-samples', type=int, default=512,
                    help='number of iwae samples (default: 512)')
parser.add_argument('--weight-avg', default='none',
                    choices=['none', 'swa', 'polyak'],
                    help='weight average method (evaluate): swa | polyak')
parser.add_argument('--weight-avg-start', type=int, default=1000,
                    help='weight average method (evaluate): swa | polyak')
parser.add_argument('--weight-avg-decay', type=float, default=0.998,
                    help='weight average method (evaluate): swa | polyak')

# final mode
parser.add_argument('--train-mode', default='train',
                    choices=['train', 'final'],
                    help='training mode: train | final')

# log
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=500,
                    help='log print-out interval (iter)')
parser.add_argument('--vis-interval', type=int, default=5000,
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

# generate cache folder
if opt.cache is None:
    opt.cache = 'experiments'
if opt.experiment is None:
    opt.experiment = '-'.join(['vae',
                               opt.dataset,
                               'm{}-mz{}-mh{}-mn{}-mnh{}-ma{}-mcl{}'.format(
                                   opt.model,
                                   opt.model_z_dim,
                                   opt.model_h_dim,
                                   opt.model_n_dim,
                                   opt.model_n_layers,
                                   opt.model_nonlin,
                                   opt.model_clip_logvar,
                                   ),
                               '{}-bt1{}'.format(opt.optimizer, opt.beta1) if opt.optimizer in ['adam', 'amsgrad'] else '{}-mt{}'.format(opt.optimizer, opt.momentum),
                               'lr{}'.format(opt.lr),
                               'wa{}{}'.format(opt.weight_avg,
                                               '-was{}-wad{}'.format(opt.weight_avg_start, opt.weight_avg_decay) if opt.weight_avg != 'none' else '',
                                               ),
                               'tbs{}'.format(opt.train_batch_size),
                               'binit{}-bfin{}-bann{:d}'.format(
                                   opt.beta_init,
                                   opt.beta_fin,
                                   int(opt.beta_annealing) if opt.beta_annealing is not None else 0),
                               'exp{}'.format(opt.exp_num if opt.exp_num else 0),
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
    opt.end_iter = load_end_iter(opt, filename='best-checkpoint.pth.tar', device=device)
else:
    opt.end_iter = None


# init dataset
train_loader, val_loader, test_loader, _ = dset.get_dataset(opt.dataset, opt.train_batch_size, opt.eval_batch_size, opt.cuda, final_mode=(opt.train_mode=='final'))


# init model
if opt.model == 'toy':
    model = net.ToyVAE(
            input_dim=opt.nchannels*opt.nheight*opt.nheight,
            h_dim=opt.model_h_dim,
            num_hidden_layers=opt.model_n_layers,
            nonlinearity=opt.model_nonlin,
            z_dim=opt.model_z_dim,
            ).to(device)
elif opt.model == 'toy-maf':
    model = net.ToyMAFVAE(
            input_dim=opt.nchannels*opt.nheight*opt.nheight,
            h_dim=opt.model_h_dim,
            num_hidden_layers=opt.model_n_layers,
            nonlinearity=opt.model_nonlin,
            z_dim=opt.model_z_dim,
            ).to(device)
elif opt.model == 'mnist':
    model = net.MNISTVAE(
            input_dim=opt.nchannels*opt.nheight*opt.nheight,
            h_dim=opt.model_h_dim,
            num_hidden_layers=opt.model_n_layers,
            nonlinearity=opt.model_nonlin,
            z_dim=opt.model_z_dim,
            do_xavier=False,
            do_m5bias=False,
            ).to(device)
elif opt.model == 'conv':
    model = net.MNISTConvVAE(
            input_height=opt.nheight,
            input_channels=opt.nchannels,
            nonlinearity=opt.model_nonlin,
            z_dim=opt.model_z_dim,
            do_xavier=False,
            do_m5bias=False,
            ).to(device)
elif opt.model == 'resconv':
    model = net.MNISTResConvVAE(
            input_height=opt.nheight,
            input_channels=opt.nchannels,
            nonlinearity=opt.model_nonlin,
            z_dim=opt.model_z_dim,
            do_m5bias=False,
            do_center=False,
            ).to(device)
elif opt.model == 'resconvct':
    model = net.MNISTResConvVAE(
            input_height=opt.nheight,
            input_channels=opt.nchannels,
            nonlinearity=opt.model_nonlin,
            z_dim=opt.model_z_dim,
            do_m5bias=False,
            do_center=False,
            ).to(device)
elif opt.model == 'auxtoy':
    model = net.ToyAuxVAE(
            input_dim=opt.nchannels*opt.nheight*opt.nheight,
            noise_dim=opt.model_n_dim,
            h_dim=opt.model_h_dim,
            num_hidden_layers=opt.model_n_layers,
            nonlinearity=opt.model_nonlin,
            enc_type='simple',
            z_dim=opt.model_z_dim,
            clip_logvar=opt.model_clip_logvar,
            ).to(device)
elif opt.model == 'auxmnist':
    model = net.MNISTAuxVAE(
            input_dim=opt.nchannels*opt.nheight*opt.nheight,
            noise_dim=opt.model_n_dim,
            h_dim=opt.model_h_dim,
            num_hidden_layers=opt.model_n_layers,
            nonlinearity=opt.model_nonlin,
            enc_type='simple',
            z_dim=opt.model_z_dim,
            do_xavier=False,
            do_m5bias=False,
            clip_logvar=opt.model_clip_logvar,
            ).to(device)
elif opt.model == 'auxconv':
    model = net.MNISTConvAuxVAE(
            input_height=opt.nheight,
            input_channels=opt.nchannels,
            z0_dim=opt.model_n_dim,
            nonlinearity=opt.model_nonlin,
            z_dim=opt.model_z_dim,
            do_xavier=False,
            do_m5bias=False,
            ).to(device)
elif opt.model == 'auxresconv':
    model = net.MNISTResConvAuxVAE(
            input_height=opt.nheight,
            input_channels=opt.nchannels,
            nonlinearity=opt.model_nonlin,
            z0_dim=opt.model_n_dim,
            z_dim=opt.model_z_dim,
            do_center=False,
            ).to(device)
elif opt.model == 'auxresconvct':
    model = net.MNISTResConvAuxVAE(
            input_height=opt.nheight,
            input_channels=opt.nchannels,
            nonlinearity=opt.model_nonlin,
            z0_dim=opt.model_n_dim,
            z_dim=opt.model_z_dim,
            do_center=True,
            ).to(device)
else:
    raise NotImplementedError
logging(str(model), path=opt.path)

''' temporary '''
_prob = get_prob_from_energy_func_for_vis(model.energy_func, num=256)
_gtlatent = get_imshow_plot(_prob, val=6 if opt.dataset in ['sbmnist', 'dbmnist', 'dbmnist-val5k'] else 4)
#img = convert_npimage_torchimage(_img)
#writer.add_image('train/latent', img.float(), 0)
''' --------- '''

# init optimizer
if opt.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=opt.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=1./4.0, patience=0, verbose=True)
elif opt.optimizer == 'adam':
    optimizer = Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    scheduler = None
elif opt.optimizer == 'amsgrad':
    optimizer = Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), amsgrad=True)
    scheduler = None
elif opt.optimizer == 'rmsprop':
    optimizer = optim.RMSprop(model.parameters(), lr=opt.lr, momentum=opt.momentum)
    scheduler = None
else:
    raise NotImplementedError('unknown optimizer: {}'.format(opt.optimizer))

# init weight avg
if opt.weight_avg == 'polyak':
    optimizer = torchcontrib.optim.Polyak(optimizer, polyak_start=opt.weight_avg_start, polyak_freq=1, polyak_decay=opt.weight_avg_decay)
elif opt.weight_avg == 'swa':
    optimizer = torchcontrib.optim.SWA(optimizer, swa_start=opt.weight_avg_start, swa_freq=1)
else:
    pass

# resume
load_checkpoint(model, optimizer, opt, filename='{}checkpoint.pth.tar'.format('final-' if opt.train_mode == 'final' else ''), device=device)


# define evaluate
def evaluate_iws(eval_loader, model, optimizer, name='valid'):
    model.eval()
    if opt.weight_avg != 'none':
        optimizer.use_buf()
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

            # forward
            _, _, _, loss, _, _ = model(eval_data)

            # logprob
            logprob = model.logprob(eval_data, sample_size=opt.iws_samples)

            # add to total loss
            cur_loss = loss.item()
            total_loss += cur_loss * batch_size
            total_elbo += -cur_loss * batch_size
            total_logprob += logprob.item() * batch_size
            num_data += batch_size

    # return
    elapsed = time.time() - start_time
    model.train()
    if opt.weight_avg != 'none':
        optimizer.use_sgd()
    return total_elbo / num_data, total_logprob / num_data, elapsed

# define train
def train(train_loader, model, optimizer, epoch, start_batch_idx=0):
    global running_train_data_iter
    model.train()
    if opt.weight_avg != 'none':
        optimizer.use_sgd()
    total_loss = 0.
    total_recon_loss = 0.
    total_kld_loss = 0.
    num_data = 0
    start_time = time.time()
    train_num_iters_per_epoch = len(train_loader)
    for _batch_idx, (train_data, _) in enumerate(train_loader):
        # init batch_idx and i_ep
        batch_idx = _batch_idx + start_batch_idx
        i_ep = (epoch-1)*train_num_iters_per_epoch + batch_idx

        # init beta
        beta = annealing_func(opt.beta_init, opt.beta_fin, opt.beta_annealing, i_ep)

        # init data
        train_data = train_data.to(device)
        batch_size = train_data.size(0)

        # init grad
        optimizer.zero_grad()

        # forward
        output, _, latent, loss, recon_loss, kld_loss = model(train_data, beta=beta)

        # scale loss
        scale = 1./float(opt.nchannels*opt.nheight*opt.nheight)
        loss = scale*loss

        # backward
        loss.backward()

        # update
        optimizer.step()

        # add to total loss
        cur_loss = loss.item() #/ scale
        total_loss += cur_loss * batch_size
        num_data += batch_size
        cur_recon_loss = recon_loss.item()
        cur_kld_loss = kld_loss.item()
        total_recon_loss += cur_recon_loss * batch_size
        total_kld_loss += cur_kld_loss * batch_size

        ''' print '''
        #if (batch_idx+1) % opt.log_interval == 0:
        if (i_ep+1) % opt.log_interval == 0:
            # set log info
            elapsed = time.time() - start_time

            # print
            #logging('| epoch {:3d} | {:5d}/{:5d} | ms/step {:5.2f} '
            logging('| iter {:d} | epoch {:3d} | {:5d}/{:5d} | ms/step {:5.2f} '
                    '| beta {:5.3f} '
                    '| loss {:5.4f} | loss (recon) {:5.4f} | loss (kld) {:5.4f} '
                    '| elbo {:5.4f} '
                    .format(
                    i_ep+1,
                    epoch,
                    batch_idx+1, train_num_iters_per_epoch,
                    elapsed * 1000 / opt.log_interval,
                    beta,
                    cur_loss,# / batch_size,
                    cur_recon_loss,# / batch_size,
                    cur_kld_loss,# / batch_size,
                    -(cur_recon_loss+cur_kld_loss),# / batch_size,
                    ),
                    path=opt.path)

            # write to tensorboard
            writer.add_scalar('{}/model/elbo/step'.format(opt.train_mode),  -(cur_recon_loss+cur_kld_loss), i_ep+1)# / batch_size, i_ep+1)
            writer.add_scalar('{}/model/loss/step'.format(opt.train_mode),  cur_loss, i_ep+1)# / batch_size, i_ep+1)
            writer.add_scalar('{}/model/recon/step'.format(opt.train_mode), cur_recon_loss, i_ep+1)# / batch_size, i_ep+1)
            writer.add_scalar('{}/model/kld/step'.format(opt.train_mode),   cur_kld_loss, i_ep+1)# / batch_size, i_ep+1)
            writer.add_scalar('{}/model/beta/step'.format(opt.train_mode),  beta, i_ep+1)

            # reset log info
            start_time = time.time()

        ''' evaluate '''
        if opt.train_mode == 'train' and opt.eval_iws_interval > 0 and (i_ep+1) % opt.eval_iws_interval == 0:
            elbo, logprob, elapsed_evaluate = evaluate_iws(val_loader, model, optimizer, name='valid')
            writer.add_scalar('val/elbo/step',        elbo,    i_ep+1)
            writer.add_scalar('val/logprob/iws/step', logprob, i_ep+1)
            logging('-' * 89, path=opt.path)
            logging('| val       '
                    '| iter {:d} | epoch {:3d} | {:5d}/{:5d} | sec/step {:5.2f} '
                    '| elbo {:.4f} '
                    '| logprob (iws) {:.4f} '
                    .format(
                    i_ep+1, epoch,
                    batch_idx+1, train_num_iters_per_epoch,
                    elapsed_evaluate,
                    elbo,
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
                    'optimizer' : optimizer.state_dict(),
                    }, opt, is_best=False, filename='best-checkpoint.pth.tar')


        ''' visualize '''
        if (i_ep+1) % opt.vis_interval == 0:
            if opt.dataset in ['swissroll', '25gaussians', 'toy3']:
                # data
                val = 6
                gens = []
                outputs = []
                latents = []
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

                    output, _, latent, _, _, _ = model(_train_data)
                    outputs += [output.detach()]
                    latents += [latent.detach()]
                gen = torch.cat(gens, dim=0).cpu()
                data = torch.cat(datas, dim=0).cpu()
                output = torch.cat(outputs, dim=0).cpu()
                latent = torch.cat(latents, dim=0).cpu()

                _data   = get_scatter_plot(data.numpy(), xlim=val, ylim=val)
                _output = get_scatter_plot(output.numpy(), xlim=val, ylim=val)
                _gen    = get_scatter_plot(gen.numpy(), xlim=val, ylim=val)
                _img = np.concatenate((_data, _output, _gen), axis=1)
                img = convert_npimage_torchimage(_img)
                writer.add_image('{}/data-recon-gen/scatter', img.float(), i_ep+1)
                _data   = get_2d_histogram_plot(data.numpy(), val=val, num=128)
                _output = get_2d_histogram_plot(output.numpy(), val=val, num=128)
                _gen    = get_2d_histogram_plot(gen.numpy(), val=val, num=128)
                _img = np.concatenate((_data, _output, _gen), axis=1)
                img = convert_npimage_torchimage(_img)
                writer.add_image('{}/data-recon-gen/heatmap', img.float(), i_ep+1)

                # latent
                val = 4
                _img = get_scatter_plot(latent.numpy(), xlim=val, ylim=val)
                img = convert_npimage_torchimage(_img)
                writer.add_image('{}/latent/scatter'.format(opt.train_mode), img.float(), i_ep+1)
                _latent = get_2d_histogram_plot(latent.numpy(), val=val, num=128)
                _img = np.concatenate((_gtlatent, _latent), axis=1)
                img = convert_npimage_torchimage(_img)
                writer.add_image('{}/latent/heatmap'.format(opt.train_mode), img.float(), i_ep+1)

            elif opt.dataset in ['sbmnist', 'dbmnist', 'dbmnist-val5k']:
                # latent
                val = 6
                latents = []
                for i in range(int(20000//opt.train_batch_size)+1):
                    try:
                        _train_data, _ = running_train_data_iter.next()
                    except:
                        running_train_data_iter = iter(train_loader)
                        _train_data, _ = running_train_data_iter.next()
                    _train_data = _train_data.to(device)
                    _batch_size = _train_data.size(0)
                    output, omu, latent, _, _, _ = model(_train_data)
                    latents += [latent.detach().view(_batch_size,-1)]
                latent = torch.cat(latents, dim=0).cpu()
                _img = get_scatter_plot(latent.numpy(), xlim=val, ylim=val)
                img = convert_npimage_torchimage(_img)
                writer.add_image('{}/latent/scatter'.format(opt.train_mode), img.float(), i_ep+1)
                _latent = get_2d_histogram_plot(latent.numpy(), val=val, num=128)
                _img = np.concatenate((_gtlatent, _latent), axis=1)
                img = convert_npimage_torchimage(_img)
                writer.add_image('{}/latent/heatmap'.format(opt.train_mode), img.float(), i_ep+1)

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
                'optimizer' : optimizer.state_dict(),
            }, opt, is_best=False, filename='{}checkpoint.pth.tar'.format('final-' if opt.train_mode == 'final' else ''))

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
                'optimizer' : optimizer.state_dict(),
            }, opt, is_best=False, filename='{}checkpoint.pth.tar'.format('final-' if opt.train_mode == 'final' else ''))

            # flush
            writer.flush()
            raise EndIterError('end of training (final)')

        ''' end epoch '''
        if batch_idx+1 == train_num_iters_per_epoch:
            # write to tensorboard
            writer.add_scalar('{}/model/elbo/epoch'.format(opt.train_mode),  -(total_recon_loss+total_kld_loss) / num_data, epoch)
            writer.add_scalar('{}/model/loss/epoch'.format(opt.train_mode),  total_loss / num_data, epoch)
            writer.add_scalar('{}/model/recon/epoch'.format(opt.train_mode), total_recon_loss / num_data, epoch)
            writer.add_scalar('{}/model/kld/epoch'.format(opt.train_mode),   total_kld_loss / num_data, epoch)
            writer.add_scalar('{}/model/beta/epoch'.format(opt.train_mode),  beta, i_ep+1)

            # flush
            writer.flush()

            return -total_loss / num_data

''' main '''
# Loop over epochs
best_val_loss = opt.best_val_loss

try:
    for epoch in range(opt.start_epoch, opt.epochs+1):
        epoch_start_time = time.time()

        # train
        train(train_loader, model, optimizer, epoch, opt.start_batch_idx)
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

        output, _, latent, _, _, _ = model(_train_data)
        outputs += [output.detach()]
        latents += [latent.detach()]
    gen = torch.cat(gens, dim=0).cpu()
    data = torch.cat(datas, dim=0).cpu()
    output = torch.cat(outputs, dim=0).cpu()
    latent = torch.cat(latents, dim=0).cpu()

    _data   = get_2d_histogram_plot(data.numpy(), val=val, num=256, use_grid=False)
    _output = get_2d_histogram_plot(output.numpy(), val=val, num=256, use_grid=False)
    _gen    = get_2d_histogram_plot(gen.numpy(), val=val, num=256, use_grid=False)
    _img = np.concatenate((_data, _output, _gen), axis=1)
    img = convert_npimage_torchimage(_img)
    writer.add_image('test/data-recon-gen/heatmap', img.float(), 0)

    # latent
    val = 4
    _latent = get_2d_histogram_plot(latent.numpy(), val=val, num=256, use_grid=False)
    _img = np.concatenate((_gtlatent, _latent), axis=1)
    img = convert_npimage_torchimage(_img)
    writer.add_image('test/latent/heatmap', img.float(), 0)

    # close writer
    writer.close()

    logging('-' * 89, path=opt.path)
    logging('-' * 89, path=opt.path)

else:
    if opt.train_mode == 'final':
        load_checkpoint(model, optimizer, opt, filename='final-checkpoint.pth.tar', device=device)
    else:
        load_checkpoint(model, optimizer, opt, filename='best-checkpoint.pth.tar', device=device)
    elbo, logprob, elapsed_evaluate = evaluate_iws(test_loader, model, optimizer, name='test')
    writer.add_scalar('test/elbo/step',        elbo,    0)
    writer.add_scalar('test/logprob/iws/step', logprob, 0)
    logging('-' * 89, path=opt.path)
    logging('| test       '
            '| sec/step {:5.2f} '
            '| elbo {:.4f} '
            '| logprob (iws) {:.4f} '
            .format(
            elapsed_evaluate,
            elbo,
            logprob,
            ),
            path=opt.path)
    logging('-' * 89, path=opt.path)

    # close writer
    writer.close()
