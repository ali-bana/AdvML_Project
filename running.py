from keras.layers import Input, Dense, Lambda
from keras.models import Model
import keras.optimizers as optimizers
import pickle
from dataset_loading import load_mnist, load_omniglot
import numpy as np
from keras.datasets import mnist
import tensorflow.keras.backend as K
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import numpy as np
from IWAE import IWAE
import dataset_loading


def scheduler(epoch, lr):
    decay = 1e-5
    return lr * 1 / (1 + decay * epoch)

def train_and_save_model(stochastic_layers, k, kind, dataset, save_directory, cache_directory, load_path=None):
    '''
    this functions trains a model and saves the weights.
    :param stochastic_layers: number of stochastic layers. 1 or 2
    :param k: K as in original paper
    :param kind: String of iwae or vae.
    :param dataset: String of mnist or omniglot
    :param save_directory: directory to save the weights
    :param cache_directory: directory where the datasets are cached
    '''
    if dataset == 'mnist':
        data = dataset_loading.load_mnist(cache_directory)
        x_train = data[:60000]

    elif dataset == 'omniglot':
        data = dataset_loading.load_omniglot(cache_directory)
        x_train = data[:24340]
    else:
        raise Exception('Invalid dataset')

    if stochastic_layers == 1:
        n_h = [50]
        n_deterministic = [200]
    elif stochastic_layers == 2:
        n_h = [100, 50]
        n_deterministic = [200, 100]
    else:
        raise Exception('Invalid number of stochastic layers')

    if kind == 'iwae':
        model = IWAE(n_deterministic, n_h, k, True, dataset, cache_directory)
    elif kind == 'vae':
        model = IWAE(n_deterministic, n_h, k, False, dataset, cache_directory)

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-4))
    if load_path != None:
        model.fit(x_train[:200], shuffle=True, batch_size=20, epochs=1)
        model.load_weights(load_path)


    if not os.path.isdir(save_directory):
        os.mkdir(save_directory)

    name = str(stochastic_layers) + 'layer_' + kind + '_' + str(k) + '_'  + dataset + '_'
    saving = tf.keras.callbacks.ModelCheckpoint(os.path.join(save_directory, name+'weights.{epoch:02d}.tf'), verbose=0, save_weights_only=True, save_freq=50)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    model.fit(x_train, shuffle=True, batch_size=20, epochs=500, callbacks=[saving, lr_scheduler])

