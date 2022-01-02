from keras.layers import Input, Dense, Lambda
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf


def sample_z_wrapper(k):
    def sample_z(args):
        local_mu, local_sigma = args
        local_mu = K.repeat(local_mu, k)
        local_sigma = K.repeat(local_sigma, k)
        eps = K.random_normal(shape=(K.shape(local_mu)[0], k, K.shape(local_mu)[2]), mean=0., stddev=1.)
        return local_mu + local_sigma * eps

    return sample_z


def loss_wrapper(k, n_h):
    log2pi = K.log(2 * np.pi)
    if n_h == 1:
        n_h1 = 50

        def loss_1h(args):
            mu, sigma, z, y_pred, y_true = args
            local_mu = K.repeat(mu, k)
            local_sigma = K.repeat(sigma, k)
            y_true = K.repeat(y_true, k)
            log_posterior = -(n_h1 / 2) * log2pi - K.sum(
                K.log(1e-8 + local_sigma) + 0.5 * K.square(z - local_mu) / K.square(1e-8 + local_sigma), axis=-1)
            log_prior = -(n_h1 / 2) * log2pi - K.sum(0.5 * K.square(z), axis=-1)
            log_bernoulli = K.sum(y_true * K.log(y_pred + 1e-8) + (1 - y_true) * K.log(1 - y_pred + 1e-8), axis=-1)
            log_weights = log_bernoulli + log_prior - log_posterior
            importance_weight = K.softmax(log_weights, axis=1)
            return -K.sum(importance_weight * log_weights, axis=-1)

        return loss_1h


class IWAE(tf.keras.Model):
    def __init__(self, k, n_h=1):
        '''
        initializes an Importance Weighted AutoEncoder. Note that the architecture of the network
        will be set equal to the one used in the original paper.
        :param k: The number of stochastic samples in each layer.
        :param n_h: The number of hidden variables (h). As in the original paper, this param can
        have values 1 or 2.
        '''
        super().__init__()
        self.k = k
        self.n_h = n_h
        if n_h == 1:
            self.encoder_deter1 = Dense(200, activation='tanh', input_shape=(None, 784))
            self.encoder_deter2 = Dense(200, activation='tanh', input_shape=(None, 200))
            self.mu = Dense(50, activation='linear', name='mu', input_shape=(None, 200))
            self.sigma = Dense(50, activation='softplus', name='sigma', input_shape=(None, 200))
            self.sampling = Lambda(sample_z_wrapper(self.k), output_shape=(self.k, 50,), name='h_1')
            self.decoder_deter1 = Dense(200, activation='relu')
            self.decoder_deter2 = Dense(200, activation='relu')
            self.decoder_out = Dense(784, activation='sigmoid')
            self.loss_layer = Lambda(loss_wrapper(self.k, self.n_h), output_shape=(None, 1))
        elif n_h == 2:
            pass
        else:
            raise Exception('n_h parameter can only be 1 or 2 not ' + str(n_h))

    def call(self, inputs, training=False):
        if self.n_h == 1:
            x = self.encoder_deter1(inputs)
            x = self.encoder_deter2(x)
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
