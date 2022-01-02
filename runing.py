import tensorflow as tf
from IWAE import IWAE, compile_loss
from dataset_loading import load_omniglot, load_mnist
import os

data = load_mnist(os.getcwd())
x_train = data[:60000]
x_test = data[60000:]

batch_size = 20
epochs = 10

mod = IWAE(5, 1)
adam = tf.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-4)
mod.compile(optimizer=adam, loss=compile_loss)
mod.fit(x_train, x_train, shuffle=True, batch_size=batch_size, epochs=epochs)
