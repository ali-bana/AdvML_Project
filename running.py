import os
import tensorflow as tf
from IWAE import IWAE
import dataset_loading
from tqdm import tqdm

def scheduler(epoch, lr):
    decay = 4 * 1e-5
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

    n_batch = x_train.shape[0] // 20


    name = str(stochastic_layers) + 'layer_' + kind + '_' + str(k) + '_'  + dataset + '_'
    saving = tf.keras.callbacks.ModelCheckpoint(os.path.join(save_directory, name+'weights.{epoch:02d}.tf'), verbose=0, save_weights_only=True, save_freq=n_batch*50)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    model.fit(x_train, shuffle=True, batch_size=20, epochs=300, callbacks=[saving, lr_scheduler])


def test_saved_model(l, load_path, cache_directory, dataset):
    if dataset == 'mnist':
        data = dataset_loading.load_mnist(cache_directory)
        x_train = data[:60000]
        x_test = data[60000:]
    elif dataset == 'omniglot':
        data = dataset_loading.load_omniglot(cache_directory)
        x_train = data[:24340]
        x_test = data[24340:]
    if l == 1:
        n_deterministic = [200]
        n_h = [50]
    elif l == 2:
        n_deterministic = [200, 100]
        n_h = [100, 50]
    else:
        raise Exception('invalid number of layers.')

    model = IWAE(n_deterministic, n_h, 5000, True, dataset, cache_directory)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-4))
    model.fit(x_train[:40], shuffle=True, batch_size=20, epochs=1)
    model.load_weights(load_path)
    temp = 0
    for x in tqdm(x_test):
        res = model(x[None, :])
        temp += res['iwae_elbo'] / x_test.shape[0]
    return temp

#%%
# import matplotlib.pyplot as plt
#
#
# e = []
# lr = 0.001
# l = []
# for epo in range(300):
#     print(lr)
#     e.append(epo)
#     lr = scheduler(epo, lr)
#     l.append(lr)
# print(l[-1])
# plt.plot(e, l)
# plt.show()