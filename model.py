import tensorflow as tf
from parameters import *

Input = tf.keras.layers.Input
Conv2D = tf.keras.layers.Conv2D
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout #keras.layers.Dropout(rate, noise_shape=None, seed=None)
Lambda = tf.keras.layers.Lambda
Flatten = tf.keras.layers.Flatten
Reshape = tf.keras.layers.Reshape
Model = tf.keras.models.Model
K = tf.keras.backend
K.set_epsilon(1e-05)

dec_in_channels = 1 #NGM SABE QUE MERDA ÉH ESTA
reshaped_dim = [-1, 7, 7, dec_in_channels] #VERIFICAR ESSES 7 AI TCHE
inputs_decoder = 18 * dec_in_channels / 2

ENC_CONV_FILTERS = [32,64,128,256]
ENC_CONV_KERNEL_SIZES = [4,4,4,4]
ENC_CONV_STRIDES = [2,2,2,2]

DEC_CONV_FILTERS = [32,64,128,256]
DEC_CONV_KERNEL_SIZES = [4,4,4,4]
DEC_CONV_STRIDES = [2,2,2,2]


def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))

def sampling(args):
    """Sample a latent vector.

    Args:
        args(tuple): Tuple containing the mean and the variance
                     to be used in the sampling.
    """
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], Z_DIM), mean=0.,stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon


enc_x = Input(shape=DEC_DIM)

enc_c0 = Conv2D(filters = ENC_CONV_FILTERS[0], kernel_size = ENC_CONV_KERNEL_SIZES[0], strides = ENC_CONV_STRIDES[0], activation=lrelu)(vae_x)
enc_d0 = Dropout(keep_prob)(enc_c0)
enc_c1 = Conv2D(filters = ENC_CONV_FILTERS[1], kernel_size = ENC_CONV_KERNEL_SIZES[1], strides = ENC_CONV_STRIDES[1], activation=lrelu)(enc_d0)
enc_d1 = Dropout(keep_prob)(enc_c1)
enc_c2 = Conv2D(filters = ENC_CONV_FILTERS[2], kernel_size = ENC_CONV_KERNEL_SIZES[2], strides = ENC_CONV_STRIDES[2], activation=lrelu)(enc_d1)
enc_d2 = Dropout(keep_prob)(enc_c2)

enc_f3 = Flatten()(enc_d2)

enc_mn = Dense(n_latent)(enc_f3)
enc_sd = Dense(n_latent)(enc_f3)

enc_z = Lambda(sampling)([enc_mn, enc_sd])

encoder = Model(enc_x, enc_z, name="encoder")
encoder.summary()




# def encoder(X_in, keep_prob):
#     activation = lrelu
#     with tf.variable_scope("encoder", reuse=None):
#         X = tf.reshape(X_in, shape=[-1, image_size[0], image_size[1], 3])
#         x = tf.layers.conv2d(X, filters=32, kernel_size=4, strides=2, padding='same', activation=activation)
#         x = tf.nn.dropout(x, keep_prob)
#         x = tf.layers.conv2d(x, filters=32, kernel_size=4, strides=2, padding='same', activation=activation)
#         x = tf.nn.dropout(x, keep_prob)
#         x = tf.layers.conv2d(x, filters=32, kernel_size=4, strides=1, padding='same', activation=activation)
#         x = tf.nn.dropout(x, keep_prob)
#         x = tf.contrib.layers.flatten(x)
#         mn = tf.layers.dense(x, units=n_latent)
#         sd       = 0.5 * tf.layers.dense(x, units=n_latent)
#         epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent]))
#         z  = mn + tf.multiply(epsilon, tf.exp(sd))
#         return z, mn, sd

def decoder(sampled_z, keep_prob):
    with tf.variable_scope("decoder", reuse=None):
        x = tf.layers.dense(sampled_z, units=inputs_decoder, activation=lrelu)
        # x = tf.layers.dense(x, units=inputs_decoder * 2 + 1, activation=lrelu)
        # x = tf.reshape(x, reshaped_dim)
        x = tf.reshape(x, [-1, 3, 3, 1])
        x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=image_size[0]*image_size[1]*3, activation=tf.nn.sigmoid)
        img = tf.reshape(x, shape=[-1, image_size[0], image_size[1], 3])
        return img
