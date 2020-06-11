'''
copied and modified from
  1) https://github.com/CW-Huang/torchkit/blob/8da6c100c48a1d1464765928fbf68fdfd99fff8a/torchkit/datasets.py
  2) https://github.com/CW-Huang/torchkit/blob/8da6c100c48a1d1464765928fbf68fdfd99fff8a/torchkit/downloader.py
'''
import os
import urllib.request
import numpy as np
import torch
floatX = 'float32'


def create(*args):
    path = '/'.join(a for a in args)
    if not os.path.isdir(path):
        os.makedirs(path)

def download_sbmnist(savedir):
    #print 'dynamically binarized mnist'
    #mnist_filenames = ['train-images-idx3-ubyte', 't10k-images-idx3-ubyte']

    #for filename in mnist_filenames:
    #    local_filename = os.path.join(savedir, filename)
    #    urllib.request.urlretrieve(
    #        "http://yann.lecun.com/exdb/mnist/{}.gz".format(
    #            filename),local_filename+'.gz')
    #    with gzip.open(local_filename+'.gz', 'rb') as f:
    #        file_content = f.read()
    #    with open(local_filename, 'wb') as f:
    #        f.write(file_content)
    #    np.savetxt(local_filename,load_mnist_images_np(local_filename))
    #    os.remove(local_filename+'.gz')

    print('download statically binarized mnist')
    subdatasets = ['train', 'valid', 'test']
    for subdataset in subdatasets:
        filename = 'binarized_mnist_{}.amat'.format(subdataset)
        url = 'http://www.cs.toronto.edu/~larocheh/'\
              'public/datasets/binarized_mnist/'\
              'binarized_mnist_{}.amat'.format(subdataset)
        local_filename = os.path.join(savedir, filename)
        urllib.request.urlretrieve(url, local_filename)

def load_sbmnist_image(root='data'):
    create(root, 'bmnist')
    droot = root+'/'+'bmnist'

    if not os.path.exists('{}/binarized_mnist_train.amat'.format(droot)):
        # download sbmnist
        download_sbmnist(droot)

    if not os.path.exists('{}/binarized_mnist_train.pt'.format(droot)):
        # Larochelle 2011
        path_tr = '{}/binarized_mnist_train.amat'.format(droot)
        path_va = '{}/binarized_mnist_valid.amat'.format(droot)
        path_te = '{}/binarized_mnist_test.amat'.format(droot)
        train_x = np.loadtxt(path_tr).astype(floatX).reshape(50000,784)
        valid_x = np.loadtxt(path_va).astype(floatX).reshape(10000,784)
        test_x = np.loadtxt(path_te).astype(floatX).reshape(10000,784)

        # save
        path_tr = '{}/binarized_mnist_train.pt'.format(droot)
        path_va = '{}/binarized_mnist_valid.pt'.format(droot)
        path_te = '{}/binarized_mnist_test.pt'.format(droot)
        train_x_pt = torch.from_numpy(train_x)
        valid_x_pt = torch.from_numpy(valid_x)
        test_x_pt  = torch.from_numpy(test_x)
        torch.save(train_x_pt, open(path_tr, 'wb'))
        torch.save(valid_x_pt, open(path_va, 'wb'))
        torch.save(test_x_pt,  open(path_te, 'wb'))

    else:
        path_tr = '{}/binarized_mnist_train.pt'.format(droot)
        path_va = '{}/binarized_mnist_valid.pt'.format(droot)
        path_te = '{}/binarized_mnist_test.pt'.format(droot)
        train_x_pt = torch.load(path_tr)
        valid_x_pt = torch.load(path_va)
        test_x_pt  = torch.load(path_te)

    return train_x_pt, valid_x_pt, test_x_pt
