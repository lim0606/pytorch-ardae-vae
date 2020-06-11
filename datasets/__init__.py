from datasets.toy import get_toy_dataset
from datasets.mnist import get_image_dataset as get_mnist_dataset
#from datasets.image import get_image_dataset

def get_dataset(dataset, train_batch_size, eval_batch_size=None, cuda=False, final_mode=False):
    if dataset in ['swissroll', '25gaussians']:
        assert final_mode == False
        return get_toy_dataset(dataset, train_batch_size, eval_batch_size, cuda)
    elif dataset in ['mnist', 'sbmnist', 'dbmnist', 'dbmnist-val5k', 'cmnist',]:
        return get_mnist_dataset(dataset, train_batch_size, eval_batch_size, cuda, final_mode=final_mode)
    #elif dataset in ['celeba',]:
    #    assert final_mode == False
    #    return get_image_dataset(dataset, train_batch_size, eval_batch_size, cuda)
    else:
        raise NotImplementationError('dataset: {}'.format(dataset))
