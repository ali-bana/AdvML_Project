from keras.layers import Input, Dense, Lambda
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf


def sample_z_wrapper(k):
    if k == -1:
        def simple_sample(args):
            local_mu, local_sigma = args
            eps = K.random_normal(shape=K.shape(local_mu), mean=0., stddev=1.)
            return local_mu + local_sigma * eps

        return simple_sample
    else:
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
    if n_h == 2:
        n_h1 = 100
        n_h2 = 50
        def loss_2h(args):
            mu_h1, sigma_h1, h1_enc, mu_h2, sigma_h2, h2, dec_mu_h1, dec_sigma_h1, h1_decoder, y_pred, y_true =  args
            mu_h1 = K.repeat(mu_h1, k)
            sigma_h1 = K.repeat(sigma_h1, k)
            y_true = K.repeat(y_true, k)
            q_h1_x = -(n_h1 / 2) * log2pi - K.sum(
                K.log(1e-8 + sigma_h1) + 0.5 * K.square(h1_enc - mu_h1) / K.square(1e-8 + sigma_h1), axis=-1)

            q_h2_h1 = -(n_h2 / 2) * log2pi - K.sum(
                K.log(1e-8 + sigma_h2) + 0.5 * K.square(h2 - mu_h2) / K.square(1e-8 + sigma_h2), axis=-1)

            p_h1_h2 = -(n_h1 / 2) * log2pi - K.sum(
                K.log(1e-8 + dec_sigma_h1) + 0.5 * K.square(h1_decoder - dec_mu_h1) / K.square(1e-8 + dec_sigma_h1), axis=-1)

            p_h2 = -(n_h2 / 2) * log2pi - K.sum(0.5 * K.square(h2), axis=-1)
            p_x_h1 = K.sum(y_true * K.log(y_pred + 1e-8) + (1 - y_true) * K.log(1 - y_pred + 1e-8), axis=-1)

            log_weights = p_x_h1 + p_h1_h2 + p_h2 - q_h1_x - q_h2_h1

            importance_weight = K.softmax(log_weights, axis=1)

            return -K.sum(importance_weight * log_weights, axis=-1)
        return  loss_2h





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
            self.sampling = Lambda(sample_z_wrapper(self.k), output_shape=(self.k, 50), name='h_1')
            self.decoder_deter1 = Dense(200, activation='tanh')
            self.decoder_deter2 = Dense(200, activation='tanh')
            self.decoder_out = Dense(784, activation='sigmoid')
        elif n_h == 2:
            self.encoder_deter1 = Dense(200, activation='tanh', input_shape=(None, 784))
            self.encoder_deter2 = Dense(200, activation='tanh', input_shape=(None, 200))
            self.mu_h1 = Dense(100, activation='linear', name='mu_h1', input_shape=(None, 200))
            self.sigma_h1 = Dense(100, activation='softplus', name='sigma_h1', input_shape=(None, 200))
            self.sampling_h1 = Lambda(sample_z_wrapper(self.k), output_shape=(self.k, 100), name='h_1')
            self.encoder_deter3 = Dense(100, activation='tanh')
            self.encoder_deter4 = Dense(100, activation='tanh')
            self.mu_h2 = Dense(50, activation='linear', name='mu_h2', input_shape=(None, None, 100))
            self.sigma_h2 = Dense(50, activation='softplus', name='sigma_h2', input_shape=(None, None, 100))
            self.sampling_h2 = Lambda(sample_z_wrapper(-1), output_shape=(self.k, 50), name='h_2')
            self.decoder_deter1 = Dense(100, activation='tanh', input_shape=(None, None, 50))
            self.decoder_deter2 = Dense(100, activation='tanh', input_shape=(None, None, 100))
            self.decoder_mu_h1 = Dense(100, activation='linear', name='decoder_mu_h1', input_shape=(None, None, 100))
            self.decoder_sigma_h1 = Dense(100, activation='softplus', name='decoder_sigma_h1',
                                          input_shape=(None, None, 100))
            self.decoder_sampling_h1 = Lambda(sample_z_wrapper(-1), name='decoder_h_1')
            self.decoder_deter3 = Dense(200, activation='tanh', input_shape=(None, None, 100))
            self.decoder_deter4 = Dense(200, activation='tanh', input_shape=(None, None, 200))
            self.decoder_out = Dense(784, activation='sigmoid', input_shape=(None, None, 200))
        else:
            raise Exception('n_h parameter can only be 1 or 2 not ' + str(n_h))

        self.loss_layer = Lambda(loss_wrapper(self.k, self.n_h), output_shape=(None, 1))

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

        if self.n_h == 2:
            x = self.encoder_deter1(inputs)
            x = self.encoder_deter2(x)
            mu_h1 = self.mu_h1(x)
            sigma_h1 = self.sigma_h1(x)
            h1_enc = self.sampling_h1(mu_h1, sigma_h1)
            x = self.encoder_deter3(h1_enc)
            x = self.encoder_deter4(x)
            mu_h2 = self.mu_h2(x)
            sigma_h2 = self.sigma_h2(x)
            h2 = self.sampling_h2([mu_h2, sigma_h2])
            x = self.decoder_deter1(h2)
            x = self.decoder_deter2(x)
            dec_mu_h1 = self.decoder_mu_h1(x)
            dec_sigma_h1 = self.decoder_sigma_h1(x)
            h1_decoder = self.decoder_sampling_h1([dec_mu_h1, dec_sigma_h1])
            x = self.decoder_deter3(h1_decoder)
            x = self.decoder_deter4(x)
            x = self.decoder_out(x)
            loss = self.loss_layer([mu_h1, sigma_h1, h1_enc,
                                    mu_h2, sigma_h2, h2,
                                    dec_mu_h1, dec_sigma_h1, h1_decoder,
                                    x, inputs])
            if training:
                return loss
            return x


def compile_loss(dummy_target, y_pred):
    return tf.squeeze(y_pred)
