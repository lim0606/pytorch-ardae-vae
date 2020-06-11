import os
from packaging import version
import numpy as np
import torch
from torchvision import datasets, transforms
from datasets.sbmnist import load_sbmnist_image


class StackedMNIST(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root=root, transform=transform, target_transform=target_transform, download=download)

    def __getitem__(self, index):
        # get indices
        mnist_size = self.__len__()
        indices = np.random.randint(mnist_size, size=2)
        index1 = indices[0] 
        index2 = indices[1]
        index3 = index

        # get item
        img1, target1 = super().__getitem__(index1)
        img2, target2 = super().__getitem__(index2)
        img3, target3 = super().__getitem__(index3)
        target = 100*target1 + 10*target2 + 1*target3
        img = torch.cat([img1, img2, img3], dim=0)
        return img, target

def get_mnist_transform(image_size=28, binary=False, center=False):
    # resize image
    if image_size != 28:
        trsfms = [transforms.Resize(image_size)]
    else:
        trsfms = []

    # default
    trsfms += [transforms.ToTensor()]

    # binary
    if binary:
        trsfms += [torch.bernoulli]

    # center
    if center:
        trsfms += [transforms.Normalize((0.5,), (0.5,))]

    # return
    return transforms.Compose(trsfms)

def get_mnist(train_batch_size, eval_batch_size, dataset, kwargs, binary=False, center=False, image_size=28, val_size=10000, final_mode=False):
    assert dataset in ['mnist', 'cmnist', 'dbmnist', 'dbmnist-val5k',]
    if dataset in ['mnist', 'cmnist', 'dbmnist', 'dbmnist-val5k']:
        DATASET = datasets.MNIST
        nclasses = 10
    else:
        raise NotImplementedError

    # init dataset (train / val)
    if version.parse(torch.__version__) <= version.parse("0.4.1"):
        raise NotImplementedError
        #train_dataset = DATASET('data/mnist', train=True, download=True, transform=get_mnist_transform(binary=binary, center=center, image_size=image_size))
        #val_dataset   = DATASET('data/mnist', train=True, download=True, transform=get_mnist_transform(binary=binary, center=center, image_size=image_size))
        #n = len(train_dataset.train_data)
        #split_filename = os.path.join('data/mnist', '{}-val{}-split.pt'.format(dataset, val_size))
        #if os.path.exists(split_filename):
        #    indices = torch.load(split_filename)
        #else:
        #    indices = torch.from_numpy(np.random.permutation(n))
        #    torch.save(indices, open(split_filename, 'wb'))
        #train_dataset.train_data   = torch.index_select(train_dataset.train_data,   0, indices[:n-val_size])
        #train_dataset.train_labels = torch.index_select(train_dataset.train_labels, 0, indices[:n-val_size])
        #val_dataset.train_data   = torch.index_select(val_dataset.train_data,   0, indices[n-val_size:])
        #val_dataset.train_labels = torch.index_select(val_dataset.train_labels, 0, indices[n-val_size:])

        ## init dataset test
        #test_dataset = DATASET('data/mnist', train=False, transform=get_mnist_transform(binary=binary, center=center, image_size=image_size))
    #else:

    # init dataset (train / val)
    train_dataset = DATASET('data', train=True, download=True, transform=get_mnist_transform(binary=binary, center=center, image_size=image_size))
    val_dataset   = DATASET('data', train=True, download=True, transform=get_mnist_transform(binary=binary, center=center, image_size=image_size)) if not final_mode else None

    # final mode
    if not final_mode:
        n = len(train_dataset.data)
        split_filename = os.path.join('data/mnist', '{}-val{}-split.pt'.format(dataset, val_size))
        if os.path.exists(split_filename):
            indices = torch.load(split_filename)
        else:
            indices = torch.from_numpy(np.random.permutation(n))
            torch.save(indices, open(split_filename, 'wb'))
        train_dataset.data    = torch.index_select(train_dataset.data,    0, indices[:n-val_size])
        train_dataset.targets = torch.index_select(train_dataset.targets, 0, indices[:n-val_size])
        val_dataset.data    = torch.index_select(val_dataset.data,    0, indices[n-val_size:])
        val_dataset.targets = torch.index_select(val_dataset.targets, 0, indices[n-val_size:])
    else:
        pass

    # init dataset test
    test_dataset = DATASET('data', train=False, transform=get_mnist_transform(binary=binary, center=center, image_size=image_size))

    # init dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=train_batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset,
            batch_size=eval_batch_size, shuffle=False, **kwargs) if not final_mode else None
    test_loader = torch.utils.data.DataLoader(test_dataset,
            batch_size=eval_batch_size, shuffle=False, **kwargs)

    # init info
    info = {}
    info['nclasses'] = nclasses

    return train_loader, val_loader, test_loader, info

def get_sbmnist(train_batch_size, eval_batch_size, dataset, kwargs, final_mode=False):
    # get bmnist
    train_data, val_data, test_data = load_sbmnist_image('data')
    train_labels, val_labels, test_labels = torch.zeros(50000).long(), torch.zeros(10000).long(), torch.zeros(10000).long()

    # final mode
    if final_mode:
        train_data = torch.cat([train_data, val_data], dim=0)
        train_labels = torch.cat([train_labels, val_labels], dim=0)
        val_data = None
        val_labels = None

    # init datasets
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    val_dataset   = torch.utils.data.TensorDataset(val_data,   val_labels) if not final_mode else None
    test_dataset  = torch.utils.data.TensorDataset(test_data,  test_labels)

    # init dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=train_batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset,
            batch_size=eval_batch_size, shuffle=False, **kwargs) if not final_mode else None
    test_loader = torch.utils.data.DataLoader(test_dataset,
            batch_size=eval_batch_size, shuffle=False, **kwargs)

    # init info
    info = {}
    info['nclasses'] = 10

    return train_loader, val_loader, test_loader, info

def get_image_dataset(dataset, train_batch_size, eval_batch_size=None, cuda=False, final_mode=False):
    # init arguments
    if eval_batch_size is None:
        eval_batch_size = train_batch_size
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

    # get dataset
    if dataset in ['mnist']:
        return get_mnist(train_batch_size, eval_batch_size, dataset, kwargs, image_size=28, binary=False, center=False, final_mode=final_mode)
    elif dataset in ['cmnist']:
        return get_mnist(train_batch_size, eval_batch_size, dataset, kwargs, image_size=28, binary=False, center=True, final_mode=final_mode)
    elif dataset in ['dbmnist']:
        return get_mnist(train_batch_size, eval_batch_size, dataset, kwargs, image_size=28, binary=True, center=False, final_mode=final_mode)
    elif dataset in ['dbmnist-val5k']:
        return get_mnist(train_batch_size, eval_batch_size, dataset, kwargs, image_size=28, binary=True, center=False, val_size=5000, final_mode=final_mode)
    elif dataset in ['sbmnist']:
        return get_sbmnist(train_batch_size, eval_batch_size, dataset, kwargs, final_mode=final_mode)
    elif dataset in ['mnist32']:
        return get_mnist(train_batch_size, eval_batch_size, dataset, kwargs, image_size=32, binary=False, center=False, final_mode=final_mode)
    else:
        raise NotImplementedError('dataset: {}'.format(dataset))
