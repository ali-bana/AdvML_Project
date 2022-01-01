!pip install wget
from keras.layers import Input, Dense, Lambda
from keras.models import Model
import keras.optimizers as optimizers
import pickle
from dataset_loading import load_mnist, load_omniglot
import numpy as np
from keras.datasets import mnist
import tensorflow.keras.backend as K
import os
import tensorflow as tf

m = 20
n_z = 50
epochs = 100
image_dim = 28
input_shape = image_dim * image_dim
deterministic_shape = 200
k = 5
log2pi = K.log(2 * np.pi)
#%%
# MNIST dataset
data = load_mnist(os.getcwd())
x_train = data[:60000]
x_test  =  data[60000:]

x_train.shape, x_test.shape



k=5
def sample_z(args):
    k = 5
    local_mu, local_sigma = args
    local_mu = K.repeat(local_mu, k)
    local_sigma = K.repeat(local_sigma, k)
    eps = K.random_normal(shape=(K.shape(local_mu)[0], k, K.shape(local_mu)[2]), mean=0., stddev=1.)
    return local_mu + local_sigma * eps

def loss_wrapper(k):
    def loss(args):
        mu, sigma, z, y_pred, y_true = args
        local_mu = K.repeat(mu, k)
        local_sigma = K.repeat(sigma, k)
        y_true = K.repeat(y_true, k)
        print('local_mu: ', K.shape(local_mu))
        print('local_sigma: ', K.shape(local_sigma))
        print('y_true: ', K.shape(y_true))
        print('y_pred: ', K.shape(y_pred))
        print('z: ', K.shape(z))
        log_posterior = -(n_z / 2) * log2pi - K.sum(K.log(1e-8 + local_sigma) + 0.5 * K.square(z - local_mu) / K.square(1e-8 + local_sigma), axis=-1)
        log_prior = -(n_z / 2) * log2pi - K.sum(0.5 * K.square(z), axis=-1)
        log_bernoulli = K.sum(y_true * K.log(y_pred + 1e-8) + (1 - y_true) * K.log(1 - y_pred + 1e-8), axis=-1)
        log_weights = log_bernoulli + log_prior - log_posterior
        importance_weight = K.softmax(log_weights, axis=1)
        return -K.sum(importance_weight * log_weights, axis=-1)
    return loss



class IWAE(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.k = 1
        self.deter1 = Dense(deterministic_shape, activation='relu', input_shape=(None,784))
        self.deter2 = Dense(deterministic_shape, activation='relu', input_shape=(None,deterministic_shape))
        self.mu = Dense(n_z, activation='linear', name='mu', input_shape=(None,deterministic_shape))
        self.sigma = Dense(n_z, activation='softplus', name='sigma', input_shape=(None,deterministic_shape))
        self.sampling = Lambda(sample_z, output_shape=(self.k, n_z,), name='z')
        self.decoder_deter1 = Dense(deterministic_shape, activation='relu')
        self.decoder_deter2 = Dense(deterministic_shape, activation='relu')
        self.decoder_out = Dense(input_shape, activation='sigmoid')
        self.loss_layer = Lambda(loss_wrapper(k), output_shape=(None,1))

    def call(self, inputs, training=False):
        x = self.deter1(inputs)
        x = self.deter2(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        z = self.sampling([mu, sigma])
        x = self.decoder_deter1(z)
        x = self.decoder_deter2(x)
        x = self.decoder_out(x)
        loss = self.loss_layer([mu, sigma, z, x, inputs])
        if training:
            return loss
        return x


def compile_loss(dummy_target, y_pred):
    return tf.squeeze(y_pred)


mod = IWAE()
# print(mod.build(input_shape=(None,784)))
# print(mod.summary())
adam = tf.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-4)
mod.compile(optimizer=adam, loss=compile_loss)
mod.fit(x_train, x_train, shuffle=True, batch_size=20, epochs=10)