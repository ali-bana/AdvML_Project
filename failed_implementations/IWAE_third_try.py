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
from tensorflow_probability import distributions as tfd
import math


def loss_wrapper(k, n_h, is_iwae=True):
    log2pi = K.log(2 * np.pi)
    if n_h == 1:
        def loss_1h(args):
            n_h1 = 50
            q_hGx, p_xGh, h, y_true = args

            p_z = tfd.Normal(0, 1)

            lpz = tf.reduce_sum(p_z.log_prob(h), axis=-1)

            lqzx = tf.reduce_sum(q_hGx.log_prob(h), axis=-1)

            lpxz = tf.reduce_sum(p_xGh.log_prob(y_true), axis=-1)

            log_weights = lpxz + lpz - lqzx
            if is_iwae:
                importance_weight = tf.stop_gradient(K.softmax(log_weights, axis=1))
                return -K.sum(importance_weight * log_weights, axis=-1)
            else:
                return -K.sum(log_weights, axis=-1)

        return loss_1h

    if n_h == 2:
        def loss_2h(args):
            n_h1 = 100
            n_h2 = 50
            q_h1Gx, q_h2Gh1, p_h1Gh2, p_xGh1, h1, h2, h1_decoder, y_true = args

            p_h2 = tfd.Normal(0, 1)

            lpz2 = tf.reduce_sum(p_h2.log_prob(h2), axis=-1)

            lqz2z1 = tf.reduce_sum(q_h2Gh1.log_prob(h2), axis=-1)

            lpz1z2 = tf.reduce_sum(p_h1Gh2.log_prob(h1_decoder), axis=-1)

            lqz1x = tf.reduce_sum(q_h1Gx.log_prob(h1), axis=-1)

            lpxz1 = tf.reduce_sum(p_xGh1.log_prob(y_true), axis=-1)

            log_weights = lpxz1 + lpz1z2 + lpz2 - lqz1x - lqz2z1

            if is_iwae:
                log_w_minus_max = log_weights - tf.reduce_max(log_weights, axis=0, keepdims=True)
                w = tf.exp(log_w_minus_max)
                w_normalized = w / tf.reduce_sum(w, axis=0, keepdims=True)
                w_normalized_stopped = tf.stop_gradient(w_normalized)
                iwae_eq14 = tf.reduce_mean(tf.reduce_sum(w_normalized_stopped * log_weights, axis=0))
                return iwae_eq14
            else:
                return -K.sum(log_weights, axis=-1)

        return loss_2h


def get_bias():
    # ---- For initializing the bias in the final Bernoulli layer for p(x|z)
    (Xtrain, ytrain), (_, _) = tf.keras.datasets.mnist.load_data()
    Ntrain = Xtrain.shape[0]

    # ---- reshape to vectors
    Xtrain = Xtrain.reshape(Ntrain, -1) / 255

    train_mean = np.mean(Xtrain, axis=0)

    bias = -np.log(1. / np.clip(train_mean, 0.001, 0.999) - 1.)

    return tf.constant_initializer(bias)


class StochasticBlock(tf.keras.Model):
    def __init__(self,
                 n_hidden,
                 n_latent,
                 **kwargs):
        super(StochasticBlock, self).__init__(**kwargs)

        self.l1 = Dense(n_hidden, activation='tanh')
        self.l2 = Dense(n_hidden, activation='tanh')
        self.mu = Dense(n_latent, activation=None)
        self.sigma = Dense(n_latent, activation=tf.keras.activations.exponential)

    def call(self, input):
        h1 = self.l1(input)
        h2 = self.l2(h1)
        mu = self.mu(h2)
        sigma = self.sigma(h2)
        return tfd.Normal(mu, sigma + 1e-6)


class IWAE(tf.keras.Model):
    def __init__(self,
                 k,
                 n_h,
                 is_iwae=True,  # if want to train a VAE should be False
                 **kwargs):
        super(IWAE, self).__init__(**kwargs)
        self.k = k
        self.n_h = n_h
        self.iwae = is_iwae
        if n_h == 1:
            n_hidden_h1 = 200
            n_h1 = 50
            self.encoder = StochasticBlock(n_hidden_h1, n_h1, name='encoder')
            self.decoder = tf.keras.Sequential([Dense(n_hidden_h1, activation="tanh", name="decoder_deter1"),
                                                Dense(n_hidden_h1, activation="tanh", name="decoder_deter2"),
                                                Dense(784, activation='sigmoid', name="decoder_out")], name='decoder')
        if n_h == 2:
            n_hidden_h1 = 200
            n_hidden_h2 = 100
            n_h1 = 100
            n_h2 = 50
            self.encoder1 = StochasticBlock(n_hidden_h1, n_h1, name='h1_encoder')
            self.encoder2 = StochasticBlock(n_hidden_h2, n_h2, name='h2_encoder')
            self.decoder_h1 = StochasticBlock(n_hidden_h2, n_h1, name='h1_decoder')
            self.decoder_out = tf.keras.Sequential \
                ([Dense(n_hidden_h1, activation="tanh", name="decoder_deter1"),
                  Dense(n_hidden_h1, activation="tanh", name="decoder_deter2"),
                  Dense(784, activation='sigmoid', name="decoder_out", bias_initializer=get_bias())],
                 name='image_decoder')

        self.loss_layer = Lambda(loss_wrapper(self.k, self.n_h, self.iwae), output_shape=(None, 1), name='loss')

    def call(self, inputs, training=False):
        if self.n_h == 1:
            q_hGx = self.encoder(inputs)
            h = q_hGx.sample(self.k)
            logits = self.decoder(h)
            p_xGh = tfd.Bernoulli(logits=logits)
            if training:
                return self.loss_layer([q_hGx, p_xGh, h, inputs])
            return logits
        else:
            q_h1Gx = self.encoder1(inputs)
            h1 = q_h1Gx.sample(self.k)

            q_h2Gh1 = self.encoder2(h1)
            h2 = q_h2Gh1.sample()

            p_h1Gh2 = self.decoder_h1(h2)
            h1_decoder = p_h1Gh2.sample()

            logits = self.decoder_out(h1)
            p_xGh1 = tfd.Bernoulli(logits=logits)
            if training:
                return self.loss_layer([q_h1Gx, q_h2Gh1, p_h1Gh2, p_xGh1, h1, h2, h1_decoder, inputs])
            return logits


def compile_loss(dummy_target, y_pred):
    return y_pred


def scheduler(epoch, lr):
    i = math.floor(math.log(epoch + 1, 3))
    return 0.001 / (10 ** i)


callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
mod = IWAE(1, 1, True)  # IWAE
# mod = IWAE(5, 2, False) #VAE
adam = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-4)
mod.compile(optimizer=adam, loss=compile_loss)
mod.fit(x_train, x_train, shuffle=True, batch_size=20, epochs=20, callbacks=[callback])

