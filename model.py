import tensorflow as tf
from parameters import *

Input = tf.keras.layers.Input
Conv2D = tf.keras.layers.Conv2D
Conv2DTranspose = tf.keras.layers.Conv2DTranspose
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

DEC_DENSE = [9, image_size(0)*image_size(1)*3]
DEC_RESHAPE = [-1, 3, 3, 1]
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

def build():
    enc_x = Input(shape=image_size)

    enc_c0 = Conv2D(filters = ENC_CONV_FILTERS[0], kernel_size = ENC_CONV_KERNEL_SIZES[0], strides = ENC_CONV_STRIDES[0], activation=lrelu)(vae_x)
    enc_d0 = Dropout(keep_prob)(enc_c0)
    enc_c1 = Conv2D(filters = ENC_CONV_FILTERS[1], kernel_size = ENC_CONV_KERNEL_SIZES[1], strides = ENC_CONV_STRIDES[1], activation=lrelu)(enc_d0)
    enc_d1 = Dropout(keep_prob)(enc_c1)
    enc_c2 = Conv2D(filters = ENC_CONV_FILTERS[2], kernel_size = ENC_CONV_KERNEL_SIZES[2], strides = ENC_CONV_STRIDES[2], activation=lrelu)(enc_d1)
    enc_d2 = Dropout(keep_prob)(enc_c2)

    enc_f = Flatten()(enc_d2)

    enc_mn = Dense(n_latent)(enc_f)
    enc_sd = Dense(n_latent)(enc_f)

    enc_z = Lambda(sampling)([enc_mn, enc_sd])

    encoder = Model(enc_x, enc_z, name="encoder")
    encoder.summary()


    dec_z = Input(shape=n_latent)

    dec_dense0 = Dense(DEC_DENSE[0])(dec_z)
    dec_r0 = Reshape(DEC_RESHAPE)(dec_d0)

    dec_c0 = Conv2DTranspose(filters = DEC_CONV_FILTERS[0], kernel_size = DEC_CONV_KERNEL_SIZES[0], strides = DEC_CONV_STRIDES[0], activation='relu')(vae_x)
    dec_d0 = Dropout(keep_prob)(dec_c0)
    dec_c1 = Conv2DTranspose(filters = DEC_CONV_FILTERS[1], kernel_size = DEC_CONV_KERNEL_SIZES[1], strides = DEC_CONV_STRIDES[1], activation='relu')(dec_d0)
    dec_d1 = Dropout(keep_prob)(dec_c1)
    dec_c2 = Conv2DTranspose(filters = DEC_CONV_FILTERS[2], kernel_size = DEC_CONV_KERNEL_SIZES[2], strides = DEC_CONV_STRIDES[2], activation='relu')(dec_d1)

    dec_f = Flatten()(dec_c2)
    dec_dense1 = DENSE(DEC_DENSE[1], activation='sigmoid')(dec_f)
    dec_x = Reshape(image_size)(dec_dense1)

    decoder = Model(dec_z, dec_x, name="decoder")
    decoder.summary()

    vae = Model(enc_x, decoder(encoder(enc_x)), name="vae")
    return vae
