import tensorflow as tf
from tensorflow_probability import distributions as tfd
import dataset_loading
import os


class StochasticBlock(tf.keras.Model):
    '''
    This class is one stochastic block consisting of two deterministic layers.
    This block's output is the distribution of the hidden values given the input.
    '''

    def __init__(self, n_deterministic, n_h, **kwargs):
        super(StochasticBlock, self).__init__(**kwargs)

        self.deter1 = tf.keras.layers.Dense(n_deterministic, activation=tf.nn.tanh)
        self.deter2 = tf.keras.layers.Dense(n_deterministic, activation=tf.nn.tanh)
        self.mu_layer = tf.keras.layers.Dense(n_h, activation=None)
        self.sigma_layer = tf.keras.layers.Dense(n_h, activation=tf.exp)

    def call(self, input):
        x = self.deter1(input)
        x = self.deter2(x)
        mu = self.mu_layer(x)
        sigma = self.sigma_layer(x)

        out_distribution = tfd.Normal(mu, sigma + 1e-6)

        return out_distribution


class Encoder(tf.keras.Model):  # This class is the encoder of VAE or IWAE.
    def __init__(self, n_deterministic, n_h, **kwargs):
        '''
        :param n_deterministic: list. Number of unit in each deterministic
        layer of the stochastic blocks
        :param n_h: list. The Dimension of the hidden variables.
        Two parameters should have the same length
        '''
        super(Encoder, self).__init__(**kwargs)
        self.n_stock = len(n_h)  # the number of the stochastic layers
        print(self.n_stock)
        if self.n_stock == 1:
            self.x_to_h = StochasticBlock(n_deterministic[0], n_h[0])
        else:
            self.x_to_h1 = StochasticBlock(n_deterministic[0], n_h[0])
            self.h1_to_h2 = StochasticBlock(n_deterministic[1], n_h[1])

    def call(self, x, k):
        if self.n_stock == 1:
            q_hGx = self.x_to_h(x)

            h = q_hGx.sample(k)

            return h, q_hGx
        else:
            q_h1Gx = self.x_to_h1(x)

            h1 = q_h1Gx.sample(k)

            q_h2Gh1 = self.h1_to_h2(h1)

            h2 = q_h2Gh1.sample()

        return h1, q_h1Gx, h2, q_h2Gh1


class Decoder(tf.keras.Model):  # This class is the encoder of VAE or IWAE.
    def __init__(self, n_deterministic, n_h, base_dir, dataset_name, **kwargs):
        '''
         :param n_deterministic: list. Number of unit in each deterministic
        layer of the stochastic blocks in reverse order. Same length to n_h.
        :param n_h: list. The Dimension of the hidden variables.
        :param base_dir: Base directory to access datasets
        :param dataset_name: Name of the Dataset being used. String
        '''
        super(Decoder, self).__init__(**kwargs)

        self.n_stock = len(n_h)  # the number of the stochastic layers
        print(self.n_stock)
        if self.n_stock == 1:
            self.decoder_out = tf.keras.Sequential(
                [tf.keras.layers.Dense(n_deterministic[0], activation=tf.nn.tanh),
                 tf.keras.layers.Dense(n_deterministic[0], activation=tf.nn.tanh),
                 tf.keras.layers.Dense(784, activation=None,
                                       bias_initializer=dataset_loading.get_bias(dataset_name, base_dir))])
        else:
            self.h2_to_h1 = StochasticBlock(n_deterministic[1], n_h[0])

            # decode z1 to x
            self.decoder_out = tf.keras.Sequential(
                [tf.keras.layers.Dense(n_deterministic[0], activation=tf.nn.tanh),
                 tf.keras.layers.Dense(n_deterministic[0], activation=tf.nn.tanh),
                 tf.keras.layers.Dense(784, activation=None,
                                       bias_initializer=dataset_loading.get_bias(dataset_name, base_dir))])

    def call(self, h1, h2=None):

        if self.n_stock == 1:
            logits = self.decoder_out(h1)
            p_xGz = tfd.Bernoulli(logits=logits)
            return logits, p_xGz
        else:
            p_h1Gh2 = self.h2_to_h1(h2)
            logits = self.decoder_out(h1)
            p_zGh1 = tfd.Bernoulli(logits=logits)
            return logits, p_zGh1, p_h1Gh2


class IWAE(tf.keras.Model):
    def __init__(self, n_deterministic, n_h, k, is_iwae=True, dataset='mnist', base_dir=os.getcwd(), **kwargs):
        super(IWAE, self).__init__(**kwargs)
        print(is_iwae)
        self.encoder = Encoder(n_deterministic, n_h)
        self.decoder = Decoder(n_deterministic, n_h, base_dir, dataset)
        self.is_IWAE = is_iwae
        self.k = k
        self.n_stock = len(n_h)

    def call(self, x):
        if self.n_stock == 1:
            h, q_hGx = self.encoder(x, self.k)

            logits, p_xGh = self.decoder(h)

            q_h = tfd.Normal(0, 1)

            log_q_h = tf.reduce_sum(q_h.log_prob(h), axis=-1)

            log_q_hGx = tf.reduce_sum(q_hGx.log_prob(h), axis=-1)

            log_p_xGh = tf.reduce_sum(p_xGh.log_prob(x), axis=-1)

            log_weights = log_p_xGh + (log_q_h - log_q_hGx)

        else:

            h1, q_h1Gx, h2, q_h2Gh1 = self.encoder(x, self.k)

            logits, p_xGh1, p_h1Gh2 = self.decoder(h1, h2)

            p_h2 = tfd.Normal(0, 1)

            log_p_h2 = tf.reduce_sum(p_h2.log_prob(h2), axis=-1)

            log_q_h2Gh1 = tf.reduce_sum(q_h2Gh1.log_prob(h2), axis=-1)

            log_p_h1Gh2 = tf.reduce_sum(p_h1Gh2.log_prob(h1), axis=-1)

            log_q_h1Gx = tf.reduce_sum(q_h1Gx.log_prob(h1), axis=-1)

            log_p_xGh1 = tf.reduce_sum(p_xGh1.log_prob(x), axis=-1)

            log_weights = log_p_xGh1 + log_p_h1Gh2 + log_p_h2 - log_q_h1Gx - log_q_h2Gh1

        vae_loss = -tf.reduce_mean(tf.reduce_mean(log_weights, axis=0), axis=-1)

        m = tf.reduce_max(log_weights, axis=0, keepdims=True)
        log_w_minus_max = log_weights - m
        w = tf.exp(log_w_minus_max)
        w_normalized = w / tf.reduce_sum(w, axis=0, keepdims=True)
        w_normalized_stopped = tf.stop_gradient(w_normalized)

        iwae_loss = -tf.reduce_mean(tf.reduce_sum(w_normalized_stopped * log_weights, axis=0))


        max = tf.reduce_max(log_weights, axis=0)
        iwae_elbo  = tf.math.log(tf.reduce_mean(tf.exp(log_weights - max), axis=0)) + max
        iwae_elbo = tf.reduce_mean(iwae_elbo, axis=-1)

        return {"vae_loss": vae_loss,
                "iwae_loss": iwae_loss,
                "logits": logits,
                'iwae_elbo': iwae_elbo}

    @tf.function
    def train_step(self, x):
        with tf.GradientTape() as tape:
            res = self.call(x)
            if self.is_IWAE:
                loss = res['iwae_loss']
            else:
                loss = res['vae_loss']

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return res

    @tf.function
    def val_step(self, x):
        return self.call(x)


