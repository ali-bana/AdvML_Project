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


def loss_wrapper(k, n_h):
    log2pi = K.log(2 * np.pi)
    if n_h == 1:
        def loss_1h(args):
            n_h1 = 50
            mu, sigma, eps, h, y_pred, y_true = args
            local_mu = K.repeat(mu, k)
            local_sigma = K.repeat(sigma, k)
            y_true = K.repeat(y_true, k)
            log_posterior = -(n_h1 / 2) * log2pi - K.sum(
                K.log(1e-8 + local_sigma) + 0.5 * K.square(eps), axis=-1)
            log_prior = -(n_h1 / 2) * log2pi - K.sum(0.5 * K.square(h), axis=-1)
            log_bernoulli = K.sum(y_true * K.log(y_pred + 1e-8) + (1 - y_true) * K.log(1 - y_pred + 1e-8), axis=-1)
            log_weights = log_bernoulli + log_prior - log_posterior
            importance_weight = tf.stop_gradient(K.softmax(log_weights, axis=1))
            return -K.sum(importance_weight * log_weights, axis=-1)

        return loss_1h

    if n_h == 2:
        def loss_2h(args):
            n_h1 = 100
            n_h2 = 50
            mu_h1_enc_rep = args[0]
            sigma_h1_enc_rep = args[1]
            eps_h1_enc = args[2]
            h1_enc = args[3]
            mu_h2_enc = args[4]
            sigma_h2_enc = args[5]
            eps_h2_enc = args[6]
            h2_enc = args[7]
            mu_h1_dec = args[8]
            sigma_h1_dec = args[9]
            eps_h1_dec = args[10]
            h1_dec = args[11]
            y_pred = args[12]
            y_true = args[13]
            y_true = K.repeat(y_true, k)

            q_h1_x = -(n_h1 / 2) * log2pi - K.sum(
                K.log(1e-8 + sigma_h1_enc_rep) + 0.5 * K.square(eps_h1_enc), axis=-1)

            q_h2_h1 = -(n_h2 / 2) * log2pi - K.sum(
                K.log(1e-8 + sigma_h2_enc) + 0.5 * K.square(eps_h2_enc), axis=-1)

            p_h2 = -(n_h2 / 2) * log2pi - K.sum(0.5 * K.square(h2_enc), axis=-1)

            p_h1_h2 = -(n_h1 / 2) * log2pi - K.sum(
                K.log(1e-8 + sigma_h1_dec) + 0.5 * K.square(eps_h1_dec), axis=-1)

            p_x_h1 = K.sum(y_true * K.log(y_pred + 1e-8) + (1 - y_true) * K.log(1 - y_pred + 1e-8), axis=-1)

            log_weights = p_h2 + p_h1_h2 + p_x_h1 - q_h1_x - q_h2_h1

            importance_weight = tf.stop_gradient(K.softmax(log_weights, axis=1))

            return -K.sum(importance_weight * log_weights, axis=-1)

        return loss_2h


# %%

class StochasticBlock(tf.keras.Model):
    def __init__(self,
                 n_hidden,
                 n_latent,
                 **kwargs):
        super(StochasticBlock, self).__init__(**kwargs)

        self.l1 = Dense(n_hidden, activation=tf.nn.tanh)
        self.l2 = Dense(n_hidden, activation=tf.nn.tanh)
        self.mu = Dense(n_latent, activation=None)
        self.sigma = Dense(n_latent, activation=tf.exp)

    def call(self, input):
        h1 = self.l1(input)
        h2 = self.l2(h1)
        mu = self.mu(h2)
        sigma = self.sigma(h2)
        return mu, sigma


class IWAE(tf.keras.Model):
    def __init__(self,
                 k,
                 n_h,
                 **kwargs):
        super(IWAE, self).__init__(**kwargs)
        self.k = k
        self.n_h = n_h
        if n_h == 1:
            n_hidden_h1 = 200
            n_h1 = 50
            self.encoder = StochasticBlock(n_hidden_h1, n_h1)
            self.decoder = tf.keras.Sequential([Dense(n_hidden_h1, activation="tanh", name="decoder_deter1"),
                                                Dense(n_hidden_h1, activation="tanh", name="decoder_deter2"),
                                                Dense(784, activation='sigmoid', name="decoder_out")])
        if n_h == 2:
            n_hidden_h1 = 200
            n_hidden_h2 = 100
            n_h1 = 100
            n_h2 = 50
            self.encoder1 = StochasticBlock(n_hidden_h1, n_h1)
            self.encoder2 = StochasticBlock(n_hidden_h2, n_h2)
            self.decoder_h1 = StochasticBlock(n_hidden_h2, n_h1)
            self.decoder_out = tf.keras.Sequential \
                ([Dense(n_hidden_h1, activation="tanh", name="decoder_deter1"),
                  Dense(n_hidden_h1, activation="tanh", name="decoder_deter2"),
                  Dense(784, activation='sigmoid', name="decoder_out")])

        self.loss_layer = Lambda(loss_wrapper(self.k, self.n_h), output_shape=(None, 1), name='loss')

    def call(self, inputs, training=False):
        if self.n_h == 1:
            mu, sigma = self.encoder(inputs)
            local_mu = K.repeat(mu, self.k)
            local_sigma = K.repeat(sigma, self.k)
            eps = K.random_normal(shape=K.shape(local_mu), mean=0., stddev=1.)
            h = local_mu + local_sigma * eps
            preds = self.decoder(h)
            if training:
                return self.loss_layer([mu, sigma, eps, h, preds, inputs])
            return preds
        else:
            mu_h1_enc, sigma_h1_enc = self.encoder1(inputs)
            mu_h1_enc_rep = K.repeat(mu_h1_enc, self.k)
            sigma_h1_enc_rep = K.repeat(sigma_h1_enc, self.k)
            eps_h1_enc = K.random_normal(shape=K.shape(mu_h1_enc_rep), mean=0., stddev=1.)
            h1_enc = mu_h1_enc_rep + sigma_h1_enc_rep * eps_h1_enc

            mu_h2_enc, sigma_h2_enc = self.encoder2(h1_enc)
            eps_h2_enc = K.random_normal(shape=K.shape(mu_h2_enc), mean=0., stddev=1.)
            h2_enc = mu_h2_enc + sigma_h2_enc * eps_h2_enc

            mu_h1_dec, sigma_h1_dec = self.decoder_h1(h2_enc)
            eps_h1_dec = K.random_normal(shape=K.shape(mu_h1_dec), mean=0., stddev=1.)
            h1_dec = mu_h1_dec + sigma_h1_dec * eps_h1_dec

            preds = self.decoder_out(h1_dec)

            if training:
                return self.loss_layer([mu_h1_enc_rep,
                                        sigma_h1_enc_rep,
                                        eps_h1_enc,
                                        h1_enc,
                                        mu_h2_enc,
                                        sigma_h2_enc,
                                        eps_h2_enc,
                                        h2_enc,
                                        mu_h1_dec,
                                        sigma_h1_dec,
                                        eps_h1_dec,
                                        h1_dec,
                                        preds,
                                        inputs])
            return preds


def compile_loss(dummy_target, y_pred):
    return tf.squeeze(y_pred)


# mod = IWAE(5, 1)
# adam = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-4)
# mod.compile(optimizer=adam, loss=compile_loss)
# mod.fit(x_train, x_train, shuffle=True, batch_size=20, epochs=1)
#

# %%
temp = tf.convert_to_tensor([[1,2,3], [4,5,6]])
print(K.repeat(temp, 7))
