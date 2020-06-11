import os
import sys
import math

import numpy as np
import sklearn.datasets
from scipy.stats import multivariate_normal

import torch
import torch.utils.data


def _normal_energy_func(x, mu=0., logvar=0.):
    energy = logvar + (x - mu)**2 / math.exp(logvar) + math.log(2.*math.pi)
    energy = 0.5 * energy
    return energy

def normal_energy_func(x, mu, logvar, ndim):
    '''
    x = b x ndim
    mu = 1 x ndim vector
    logvar = scalar
    ndim = scalar
    '''
    assert x.dim() == 2
    assert x.size(1) == ndim
    batch_size = x.size(0)
    x = x.view(batch_size, -1)
    mu = mu.view(1, ndim).contiguous()

    energy = torch.sum(_normal_energy_func(x, mu.expand_as(x), logvar), dim=1, keepdim=True)
    return energy

def normal_logprob(x, mu, std, ndim):
    '''
    Inputs:⋅
        x: b1 x ndim⋅
        mu: 1 x ndim
        logvar: scalar
    Outputs:
        prob: b1 x 1
    '''
    var = std**2
    logvar = math.log(var)
    logprob = - normal_energy_func(x, mu, logvar, ndim)
    return logprob

def normal_prob(x, mu, std, ndim):
    logprob = normal_logprob(x, mu, std, ndim)
    prob = torch.exp(logprob)
    return prob


# swissroll
def get_swissroll(num_data):
    '''
    copied and modified from https://github.com/caogang/wgan-gp/blob/ae47a185ed2e938c39cf3eb2f06b32dc1b6a2064/gan_toy.py#L153
    '''
    # data
    data = sklearn.datasets.make_swiss_roll(
        n_samples=num_data,
        noise=0.75, #0.25
    )
    data = data[0]
    data = data.astype('float32')[:, [0, 2]]
    data /= 3. #/= 7.5  # stdev plus a little

    # target
    target = np.zeros(num_data)

    # convert to torch tensor
    data = torch.from_numpy(data)
    target = torch.from_numpy(target)

    return data, target, None, None

# exp1: single gaussian
def exp1(num_data=1000):

    var = 1.0
    std = var**(0.5)
    N  = 1
    if num_data % N != 0:
        raise ValueError('num_data should be multiple of {} (num_data = {})'.format(N, num_data))

    # data and label
    mu = np.array([[0, 0]], dtype=np.float)
    mu = torch.from_numpy(mu)
    num_data_per_mixture = num_data // N
    sigma = math.sqrt(var)
    x = torch.zeros(num_data, 2)
    label = torch.LongTensor(num_data).zero_()
    for i in range(N):
        xx = x[i*num_data_per_mixture:(i+1)*num_data_per_mixture, :]
        xx.copy_(torch.cat(
                 (torch.FloatTensor(num_data_per_mixture).normal_(mu[i,0], sigma).view(num_data_per_mixture, 1),
                  torch.FloatTensor(num_data_per_mixture).normal_(mu[i,1], sigma).view(num_data_per_mixture, 1)), 1))
        label[i*num_data_per_mixture:(i+1)*num_data_per_mixture] = i

    # classifier
    def classifier(x):
        numer = [(1.0 / float(N)) * normal_prob(x, mu[i, :], std, 2) for i in range(N)]
        numer = torch.cat(numer, dim=1)
        denom = torch.sum(numer, dim=1, keepdim=True)
        prob = numer / (denom + 1e-10)
        pred = prob.argmax(dim=1, keepdim=True)
        return pred

    # pdf
    def pdf(x):
        prob = 0
        for i in range(N):
            _mu = mu[i, :]
            prob += (1.0 / float(N)) * normal_prob(x, _mu, std, 2)
        return prob

    def logpdf(x):
        return torch.log((pdf(x) + 1e-10))

    #return x, label, sumloglikelihood
    #return x, label, logpdf, classifier#, sumloglikelihood
    return x, label, None, None#, sumloglikelihood

# mixture of three Gaussians
def exp3(num_data=1000):
    if num_data % 4 != 0:
        raise ValueError('num_data should be multiple of 4. num_data = {}'.format(num_data))

    center = 2
    sigma = 0.5 #math.sqrt(3)
    num_of_modes = 3

    # init data⋅
    d1x = torch.FloatTensor(num_data//num_of_modes, 1) #//4, 1)
    d1y = torch.FloatTensor(num_data//num_of_modes, 1) #//4, 1)
    d1x.normal_(center, sigma * 1) #3)
    d1y.normal_(center, sigma * 1)

    #d2x = torch.FloatTensor(num_data//4, 1)
    #d2y = torch.FloatTensor(num_data//4, 1)
    #d2x.normal_(-center, sigma * 1)
    #d2y.normal_(center,  sigma * 1) #3)

    d3x = torch.FloatTensor(num_data//num_of_modes, 1)
    d3y = torch.FloatTensor(num_data//num_of_modes, 1)
    d3x.normal_(center,  sigma * 1) #3)
    d3y.normal_(-center, sigma * 1) #2)

    d4x = torch.FloatTensor(num_data//num_of_modes, 1) #//4, 1)
    d4y = torch.FloatTensor(num_data//num_of_modes, 1) #//4, 1)
    d4x.normal_(-center, sigma * 1) #2)
    d4y.normal_(-center, sigma * 1) #2)

    d1 = torch.cat((d1x, d1y), 1)
    #d2 = torch.cat((d2x, d2y), 1)
    d3 = torch.cat((d3x, d3y), 1)
    d4 = torch.cat((d4x, d4y), 1)

    #d = torch.cat((d1, d2, d3, d4), 0)
    d = torch.cat((d1, d3, d4), 0)

    # label
    label = torch.LongTensor((num_data//num_of_modes)*num_of_modes).zero_()
    #for i in range(4):
    #    label[i*(num_data//4):(i+1)*(num_data//4)] = i
    for i in range(num_of_modes):
        label[i*(num_data//num_of_modes):(i+1)*(num_data//num_of_modes)] = i

    # shuffle
    shuffle = torch.randperm(d.size()[0])
    d = torch.index_select(d, 0, shuffle)
    label = torch.index_select(label, 0, shuffle)

    # pdf
    #rv1 = multivariate_normal([ center,  center], [[math.pow(sigma * 3, 2), 0.0], [0.0, math.pow(sigma * 1, 2)]])
    #rv2 = multivariate_normal([-center,  center], [[math.pow(sigma * 1, 2), 0.0], [0.0, math.pow(sigma * 3, 2)]])
    #rv3 = multivariate_normal([ center, -center], [[math.pow(sigma * 3, 2), 0.0], [0.0, math.pow(sigma * 2, 2)]])
    #rv4 = multivariate_normal([-center, -center], [[math.pow(sigma * 2, 2), 0.0], [0.0, math.pow(sigma * 2, 2)]])
    rv1 = multivariate_normal([ center,  center], [[math.pow(sigma * 1, 2), 0.0], [0.0, math.pow(sigma * 1, 2)]])
    rv3 = multivariate_normal([ center, -center], [[math.pow(sigma * 1, 2), 0.0], [0.0, math.pow(sigma * 1, 2)]])
    rv4 = multivariate_normal([-center, -center], [[math.pow(sigma * 1, 2), 0.0], [0.0, math.pow(sigma * 1, 2)]])

    def pdf(x):
        #prob = 0.25 * rv1.pdf(x) + 0.25 * rv2.pdf(x) + 0.25 * rv3.pdf(x) + 0.25 * rv4.pdf(x)
        prob = 1./float(num_of_modes) (rv1.pdf(x) + rv3.pdf(x) + rv4.pdf(x))
        return prob

    def sumloglikelihood(x):
        return np.sum(np.log((pdf(x) + 1e-10)))

    #return d, label, sumloglikelihood
    return d, label, None, None

# exp4: grid shapes
def exp4(num_data=1000):

    var = 0.1
    std = var**(0.5)
    max_x = 4 #21
    max_y = 4 #21
    min_x = -max_x
    min_y = -max_y
    n = 5

    # init
    nx, ny = (n, n)
    x = np.linspace(min_x, max_x, nx)
    y = np.linspace(min_y, max_y, ny)
    xv, yv = np.meshgrid(x, y)
    N  = xv.size 
    if num_data % N != 0:
        raise ValueError('num_data should be multiple of {} (num_data = {})'.format(N, num_data))

    # data and label
    mu = np.concatenate((xv.reshape(N,1), yv.reshape(N,1)), axis=1)
    mu = torch.FloatTensor(mu)
    num_data_per_mixture = num_data // N
    sigma = math.sqrt(var)
    x = torch.zeros(num_data, 2)
    label = torch.LongTensor(num_data).zero_()
    for i in range(N):
        xx = x[i*num_data_per_mixture:(i+1)*num_data_per_mixture, :]
        xx.copy_(torch.cat(
                 (torch.FloatTensor(num_data_per_mixture).normal_(mu[i,0], sigma).view(num_data_per_mixture, 1),
                  torch.FloatTensor(num_data_per_mixture).normal_(mu[i,1], sigma).view(num_data_per_mixture, 1)), 1))
        label[i*num_data_per_mixture:(i+1)*num_data_per_mixture] = i 

    # classifier
    def classifier(x):
        numer = [(1.0 / float(N)) * normal_prob(x, mu[i, :], std, 2) for i in range(N)]
        numer = torch.cat(numer, dim=1)
        denom = torch.sum(numer, dim=1, keepdim=True)
        prob = numer / (denom + 1e-10)
        pred = prob.argmax(dim=1, keepdim=True)
        return pred

    # pdf
    def pdf(x):
        prob = 0
        for i in range(N):
            _mu = mu[i, :]
            prob += (1.0 / float(N)) * normal_prob(x, _mu, std, 2)
        return prob

    def logpdf(x):
        return torch.log((pdf(x) + 1e-10))

    #return x, label, sumloglikelihood
    #return x, label, logpdf, classifier#, sumloglikelihood
    return x, label, None, None

def get_toy_data(name, num_data):
    if name == 'swissroll':
        return get_swissroll(num_data)
    elif name == 'gaussian':
        return exp1(num_data)
    elif name == '25gaussians':
        return exp4(num_data)
    elif name == 'toy3':
        return exp3(num_data)
    else:
        raise NotImplementedError('no toy data: {}'.format(name))

def generate_data(name, num_train_samples=2000000, num_test_samples=20000, num_val_samples=2000):
    path = 'data/toy'
    os.system('mkdir -p {}'.format(path))

    # generate
    train_data_tensor, train_target_tensor, logpdf, classifier = get_toy_data(name, num_train_samples)
    val_data_tensor, val_target_tensor, _, _ = get_toy_data(name, num_val_samples)
    test_data_tensor, test_target_tensor, _, _ = get_toy_data(name, num_test_samples)

    # save
    with open(os.path.join(path, '{}.pt'.format(name)), 'wb') as f:
        torch.save({
            'train': (train_data_tensor, train_target_tensor),
            'val': (val_data_tensor, val_target_tensor),
            'test': (test_data_tensor, test_target_tensor),
            'logpdf': logpdf,
            'classifier': classifier,
            }, f)

    # return
    return (logpdf, classifier,
            train_data_tensor, train_target_tensor,
            val_data_tensor, val_target_tensor,
            test_data_tensor, test_target_tensor,
            )

'''
get dataset with name
'''
def get_toy_dataset_with_name(name, train_batch_size, eval_batch_size, kwargs):
    path = 'data/toy'
    filename = os.path.join(path, '{}.pt'.format(name))
    if os.path.exists(filename):
        data = torch.load(filename)
        logpdf = data['logpdf']
        classifier = data['classifier']
        train_data_tensor, train_target_tensor = data['train']
        val_data_tensor, val_target_tensor = data['val']
        test_data_tensor, test_target_tensor = data['test']
    else:
        (logpdf, classifier,
            train_data_tensor, train_target_tensor,
            val_data_tensor, val_target_tensor,
            test_data_tensor, test_target_tensor,
            ) = generate_data(name)

    # init dataset (train / val)
    train_dataset = torch.utils.data.TensorDataset(train_data_tensor, train_target_tensor.long())
    val_dataset = torch.utils.data.TensorDataset(val_data_tensor, val_target_tensor.long())
    test_dataset = torch.utils.data.TensorDataset(test_data_tensor, test_target_tensor.long())

    # init dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=train_batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset,
            batch_size=train_batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset,
            batch_size=eval_batch_size, shuffle=False, **kwargs)

    # init info
    info = {}
    info['nclasses'] = len(torch.unique(train_target_tensor))
    info['classifier'] = classifier
    info['logpdf'] = logpdf

    return train_loader, val_loader, test_loader, info


'''
get dataset
'''
def get_toy_dataset(dataset, train_batch_size, eval_batch_size=None, cuda=False):
    # init arguments
    if eval_batch_size is None:
        eval_batch_size = train_batch_size
    kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}

    # get dataset
    if dataset in ['swissroll', 'toy3', '25gaussians', 'gaussian']:
        return get_toy_dataset_with_name(dataset, train_batch_size, eval_batch_size, kwargs)
    else:
        raise NotImplementedError('dataset: {}'.format(dataset))
