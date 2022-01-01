import os
import tensorflow_datasets as tfds
import numpy as np
import scipy
import scipy.io
import wget

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
                          os.path.join(os.getcwd(),'data', 'chardata.mat'))
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
    print(data.dtype)
    np.save(os.path.join(base_dir, 'data', 'mnistBinerized.npy'), data)
    return data


d = load_omniglot(os.getcwd())
print(d.dtype)