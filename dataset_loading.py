import os
import tensorflow_datasets as tfds
import numpy as np
import scipy
import scipy.io
import wget
import tensorflow as tf

'''This file contains two simple functions for loading the MNIST anf Omnilog datasets.
Inputs are the base url to save the data. They make base_dir/data subdirectory and cache the 
datasets in it. They return the binarized array of all data in each dataset. The mnist is 
downloaded using tensorflow and the 28*28 omniglot is downloaded from the data folder in the
implementation of the paper by authors.
'''


def download_and_save(url, save_path):
    wget.download(url, save_path)


def load_omniglot(base_dir):
    '''
    downloads the dataset and returns the binarized data.
    :param base_dir: String. The base directory to cache datasets
    :return: a 32415*728 numpy array
    '''
    if os.path.isfile(os.path.join(base_dir, 'data', 'omniglotBinerized.npy')):
        data = np.load(os.path.join(base_dir, 'data', 'omniglotBinerized.npy'))
        return data
    if not os.path.isdir(os.path.join(base_dir, 'data')):
        os.mkdir(os.path.join(base_dir, 'data'))
    if not os.path.isfile(os.path.join(base_dir, 'data', 'chardata.mat')):
        download_and_save('https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat',
                          os.path.join(base_dir,'data', 'chardata.mat'))
    d = scipy.io.loadmat(os.path.join(base_dir, 'data', 'chardata.mat'))
    data = np.concatenate([d['data'].T.astype('float64'), d['testdata'].T.astype('float64')])
    data = np.random.binomial(1, data).astype('float32')
    np.save(os.path.join(base_dir, 'data', 'omniglotBinerized.npy'), data)
    return data


def load_mnist(base_dir):
    '''
        downloads the dataset and returns the binarized data.
        :param base_dir: String. The base directory to cache datasets
        :return: a 70000*728 numpy array
        '''
    if os.path.isfile(os.path.join(base_dir, 'data', 'mnistBinerized.npy')):
        data = np.load(os.path.join(base_dir, 'data', 'mnistBinerized.npy'))
        return data
    if not os.path.isdir(os.path.join(base_dir, 'data')):
        os.mkdir(os.path.join(base_dir, 'data'))
    d = tfds.load('mnist', shuffle_files=True, data_dir=os.path.join(base_dir, 'data'))
    data = np.concatenate([np.stack([e['image'] for e in tfds.as_numpy(d['train'])]),
                           np.stack([e['image'] for e in tfds.as_numpy(d['test'])])]).reshape([-1, 28 * 28]).astype(
        'float64')
    data /= 255
    data = np.random.binomial(1, data).astype('float32')
    np.save(os.path.join(base_dir, 'data', 'mnistBinerized.npy'), data)
    return data


def load_pure_dataset(name, base_dir):
    if name == 'mnist':
        if not os.path.isdir(os.path.join(base_dir, 'data')):
            os.mkdir(os.path.join(base_dir, 'data'))
        d = tfds.load('mnist', shuffle_files=True, data_dir=os.path.join(base_dir, 'data'))
        data = np.concatenate([np.stack([e['image'] for e in tfds.as_numpy(d['train'])]),
                               np.stack([e['image'] for e in tfds.as_numpy(d['test'])])]).reshape([-1, 28 * 28]).astype(
            'float64')
        return data / 255
    elif name == 'omniglot':
        if not os.path.isdir(os.path.join(base_dir, 'data')):
            os.mkdir(os.path.join(base_dir, 'data'))
        if not os.path.isfile(os.path.join(base_dir, 'data', 'chardata.mat')):
            download_and_save('https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat',
                              os.path.join(base_dir, 'data', 'chardata.mat'))
        d = scipy.io.loadmat(os.path.join(base_dir, 'data', 'chardata.mat'))
        data = np.concatenate([d['data'].T.astype('float64'), d['testdata'].T.astype('float64')])
        return data
    raise Exception('Invalid dataset')


def get_bias(dataset_name, base_dir):
    '''
    this function is used for initialization of the last layer bias.
    :param dataset_name: the name of the dataset currently=[omniglot or mnist]
    :param base_dir: the base directory as used in dataset_loading file
    :return: the initial bias
    '''
    x_train = load_pure_dataset(dataset_name, base_dir)
    dim = x_train.shape[1]
    x_train = x_train.reshape(dim, -1) / 255
    train_mean = np.mean(x_train, axis=1)
    bias = -np.log(1. / np.clip(train_mean, 0.001, 0.999) - 1.)
    return tf.constant_initializer(bias)
