import tensorflow as tf
from parameters import *

Input = tf.keras.layers.Input
Conv2D = tf.keras.layers.Conv2D
Conv2DTranspose = tf.keras.layers.Conv2DTranspose
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Lambda = tf.keras.layers.Lambda
Flatten = tf.keras.layers.Flatten
Reshape = tf.keras.layers.Reshape
Model = tf.keras.models.Model
K = tf.keras.backend

to_categorical = tf.keras.utils.to_categorical
losses = tf.keras.losses
optimizers = tf.keras.optimizers


ENC_CONV_FILTERS = [64, 32, 16, 8, 12, 12]
ENC_CONV_KERNEL_SIZES = [4, 4, 2, 2, 2, 2]
ENC_CONV_STRIDES = [2, 2, 1, 1, 1, 1]

DEC_DENSE = [4, image_size[0]*image_size[1]*image_size[2]]
DEC_RESHAPE = [2, 2, 1]
DEC_CONV_FILTERS = [12, 12, 8, 16, 32, 64]
DEC_CONV_KERNEL_SIZES = [2, 2, 2, 2, 4, 4]
DEC_CONV_STRIDES = [1, 1, 1, 1, 2, 2]


def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))

def sampling(args):
    """Sample a latent vector.

    Args:
        args(tuple): Tuple containing the mean and the variance
                     to be used in the sampling.
    """
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], n_latent), mean=0.,stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

def build():

    # --------- ENCODER ---------
    enc_x = Input(shape=image_size)

    enc_c0 = Conv2D(filters = ENC_CONV_FILTERS[0], kernel_size = ENC_CONV_KERNEL_SIZES[0], strides = ENC_CONV_STRIDES[0], activation=lrelu)(enc_x)
    # enc_d0 = Dropout(keep_prob)(enc_c0)
    enc_c1 = Conv2D(filters = ENC_CONV_FILTERS[1], kernel_size = ENC_CONV_KERNEL_SIZES[1], strides = ENC_CONV_STRIDES[1], activation=lrelu)(enc_c0)
    # enc_d1 = Dropout(keep_prob)(enc_c1)
    enc_c2 = Conv2D(filters = ENC_CONV_FILTERS[2], kernel_size = ENC_CONV_KERNEL_SIZES[2], strides = ENC_CONV_STRIDES[2], activation=lrelu)(enc_c1)
    # enc_d2 = Dropout(keep_prob)(enc_c2)
    enc_c3 = Conv2D(filters = ENC_CONV_FILTERS[3], kernel_size = ENC_CONV_KERNEL_SIZES[3], strides = ENC_CONV_STRIDES[3], activation=lrelu)(enc_c2)
    # enc_d3 = Dropout(keep_prob)(enc_c3)
    # enc_c4 = Conv2D(filters = ENC_CONV_FILTERS[4], kernel_size = ENC_CONV_KERNEL_SIZES[4], strides = ENC_CONV_STRIDES[4], activation=lrelu)(enc_d3)
    # enc_d4 = Dropout(keep_prob)(enc_c4)
    # enc_c5 = Conv2D(filters = ENC_CONV_FILTERS[5], kernel_size = ENC_CONV_KERNEL_SIZES[5], strides = ENC_CONV_STRIDES[5], activation=lrelu)(enc_d4)
    # enc_d5 = Dropout(keep_prob)(enc_c5)

    enc_f = Flatten()(enc_c3)

    enc_mn = Dense(n_latent)(enc_f)
    enc_sd = Dense(n_latent)(enc_f)

    enc_z = Lambda(sampling)([enc_mn, enc_sd])

    encoder = Model(enc_x, enc_z, name="encoder")
    encoder.summary()


    # --------- DECODER ---------
    dec_z = Input(shape=(n_latent,))

    dec_dense0 = Dense(DEC_DENSE[0])(dec_z)
    dec_r0 = Reshape(DEC_RESHAPE)(dec_dense0)

    # dec_c0 = Conv2DTranspose(filters = DEC_CONV_FILTERS[0], kernel_size = DEC_CONV_KERNEL_SIZES[0], strides = DEC_CONV_STRIDES[0], activation='relu')(dec_r0)
    # dec_d0 = Dropout(keep_prob)(dec_c0)
    # dec_c1 = Conv2DTranspose(filters = DEC_CONV_FILTERS[1], kernel_size = DEC_CONV_KERNEL_SIZES[1], strides = DEC_CONV_STRIDES[1], activation='relu')(dec_d0)
    # dec_d1 = Dropout(keep_prob)(dec_c1)
    dec_c2 = Conv2DTranspose(filters = DEC_CONV_FILTERS[2], kernel_size = DEC_CONV_KERNEL_SIZES[2], strides = DEC_CONV_STRIDES[2], activation='relu')(dec_r0)
    # dec_d2 = Dropout(keep_prob)(dec_c2)
    dec_c3 = Conv2DTranspose(filters = DEC_CONV_FILTERS[3], kernel_size = DEC_CONV_KERNEL_SIZES[3], strides = DEC_CONV_STRIDES[3], activation='relu')(dec_c2)
    # dec_d3 = Dropout(keep_prob)(dec_c3)
    dec_c4 = Conv2DTranspose(filters = DEC_CONV_FILTERS[4], kernel_size = DEC_CONV_KERNEL_SIZES[4], strides = DEC_CONV_STRIDES[4], activation='relu')(dec_c3)
    # dec_d4 = Dropout(keep_prob)(dec_c4)
    dec_c5 = Conv2DTranspose(filters = DEC_CONV_FILTERS[5], kernel_size = DEC_CONV_KERNEL_SIZES[5], strides = DEC_CONV_STRIDES[5], activation='relu')(dec_c4)
    # dec_d5 = Dropout(keep_prob)(dec_c5)


    dec_f = Flatten()(dec_c5)
    dec_dense1 = Dense(DEC_DENSE[1], activation='sigmoid')(dec_f)
    dec_x = Reshape(image_size)(dec_dense1)

    decoder = Model(dec_z, dec_x, name="decoder")
    decoder.summary()


    # --------- FULL VAE ---------
    vae = Model(enc_x, decoder(encoder(enc_x)), name="vae")


    # --------- TRAINING DEFINITIONS ---------
    def vae_loss(y_true, y_pred):
        y_true_flat = K.flatten(y_true)
        y_pred_flat = K.flatten(y_pred)

        recon = 10*K.mean(K.square(y_true_flat - y_pred_flat))
        # kl = - 0.5 * K.sum(1 + enc_sd - K.square(enc_mn) - K.exp(enc_sd))
        kl = 0
        return K.mean(recon + kl)

    Adam = optimizers.Adam(lr=0.0005)
    vae.compile(optimizer=Adam, loss=vae_loss, metrics=['acc'])

    return vae, encoder, decoder
