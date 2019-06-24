import tensorflow as tf
from parameters import *
from tensorflow.keras.applications.resnet50 import ResNet50

Input = tf.keras.layers.Input
Conv2D = tf.keras.layers.Conv2D
Conv2DTranspose = tf.keras.layers.Conv2DTranspose
MaxPooling2D = tf.keras.layers.MaxPooling2D
UpSampling2D = tf.keras.layers.UpSampling2D
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Lambda = tf.keras.layers.Lambda
Flatten = tf.keras.layers.Flatten
Reshape = tf.keras.layers.Reshape
Model = tf.keras.models.Model
K = tf.keras.backend
K.set_epsilon(1e-05)
BatchNormalization = tf.keras.layers.BatchNormalization

to_categorical = tf.keras.utils.to_categorical
losses = tf.keras.losses
optimizers = tf.keras.optimizers

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

ENC_CONV_FILTERS =      [64, 128, 256]
ENC_CONV_KERNEL_SIZES = [5, 1]
ENC_CONV_STRIDES =      [1, 2, 2, 1, 1, 2, 1, 2]

DEC_DENSE = [8*8*256, image_size[0]*image_size[1]*image_size[2]]
DEC_RESHAPE = [8, 8, 256]

DEC_CONV_FILTERS = [256, 128, 64, image_size[2]]
DEC_CONV_KERNEL_SIZES = [5, 4]
DEC_CONV_STRIDES = [2, 1, 2, 1, 2, 2, 1]


def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], n_latent), mean=0.,stddev=1)
    return z_mean + K.exp(z_log_var / 2) * epsilon

def build():

    # --------- ENCODER ---------
    # enc_x = Input(shape=image_size)
    # enc_h = BatchNormalization()(enc_x)
    # enc_h = Conv2D(filters = ENC_CONV_FILTERS[0], kernel_size = ENC_CONV_KERNEL_SIZES[0], strides = ENC_CONV_STRIDES[0], activation=lrelu)(enc_x)
    # enc_h = BatchNormalization()(enc_h)
    # enc_h = Conv2D(filters = ENC_CONV_FILTERS[0], kernel_size = ENC_CONV_KERNEL_SIZES[0], strides = ENC_CONV_STRIDES[1], activation=lrelu)(enc_h)
    # enc_h = BatchNormalization()(enc_h)
    # enc_h = Conv2D(filters = ENC_CONV_FILTERS[0], kernel_size = ENC_CONV_KERNEL_SIZES[0], strides = ENC_CONV_STRIDES[2], activation=lrelu)(enc_h)
    # enc_h = BatchNormalization()(enc_h)
    # enc_h = Conv2D(filters = ENC_CONV_FILTERS[0], kernel_size = ENC_CONV_KERNEL_SIZES[0], strides = ENC_CONV_STRIDES[3], activation=lrelu)(enc_h)
    # enc_h = BatchNormalization()(enc_h)
    # enc_h = Conv2D(filters = ENC_CONV_FILTERS[0], kernel_size = ENC_CONV_KERNEL_SIZES[0], strides = ENC_CONV_STRIDES[4], activation=lrelu)(enc_h)
    # enc_h = BatchNormalization()(enc_h)
    # enc_h = Conv2D(filters = ENC_CONV_FILTERS[1], kernel_size = ENC_CONV_KERNEL_SIZES[0], strides = ENC_CONV_STRIDES[5], activation=lrelu)(enc_h)
    # enc_h = BatchNormalization()(enc_h)
    # enc_h = Conv2D(filters = ENC_CONV_FILTERS[1], kernel_size = ENC_CONV_KERNEL_SIZES[0], strides = ENC_CONV_STRIDES[6], activation=lrelu)(enc_h)
    # enc_h = BatchNormalization()(enc_h)
    # enc_h = Conv2D(filters = ENC_CONV_FILTERS[2], kernel_size = ENC_CONV_KERNEL_SIZES[1], strides = ENC_CONV_STRIDES[7], activation=lrelu)(enc_h)
    # enc_h = BatchNormalization()(enc_h)

    resnet = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    enc_x = resnet.input

    enc_h = resnet.layers[-1].output

    enc_h = Flatten()(enc_h)
    enc_mn = Dense(n_latent)(enc_h)
    enc_sd = Dense(n_latent)(enc_h)

    enc_z = Lambda(sampling)([enc_mn, enc_sd])

    encoder = Model(enc_x, enc_z, name="encoder")
    encoder.summary()


    # --------- DECODER ---------
    dec_z = Input(shape=(n_latent,))

    dec_dense0 = Dense(DEC_DENSE[0])(dec_z)
    dec_r0 = Reshape(DEC_RESHAPE)(dec_dense0)
    dec_h = dec_r0
    dec_h = Conv2DTranspose(filters = DEC_CONV_FILTERS[0], kernel_size = DEC_CONV_KERNEL_SIZES[0], strides = DEC_CONV_STRIDES[0], activation=lrelu)(dec_h)
    dec_h = BatchNormalization()(dec_h)
    dec_h = Conv2DTranspose(filters = DEC_CONV_FILTERS[0], kernel_size = DEC_CONV_KERNEL_SIZES[0], strides = DEC_CONV_STRIDES[1], activation=lrelu)(dec_h)
    dec_h = BatchNormalization()(dec_h)
    dec_h = Conv2DTranspose(filters = DEC_CONV_FILTERS[0], kernel_size = DEC_CONV_KERNEL_SIZES[0], strides = DEC_CONV_STRIDES[2], activation=lrelu)(dec_h)
    dec_h = BatchNormalization()(dec_h)
    dec_h = Conv2DTranspose(filters = DEC_CONV_FILTERS[0], kernel_size = DEC_CONV_KERNEL_SIZES[0], strides = DEC_CONV_STRIDES[3], activation=lrelu)(dec_h)
    dec_h = BatchNormalization()(dec_h)
    dec_h = Conv2DTranspose(filters = DEC_CONV_FILTERS[1], kernel_size = DEC_CONV_KERNEL_SIZES[0], strides = DEC_CONV_STRIDES[4], activation=lrelu)(dec_h)
    dec_h = BatchNormalization()(dec_h)
    dec_h = Conv2DTranspose(filters = DEC_CONV_FILTERS[2], kernel_size = DEC_CONV_KERNEL_SIZES[0], strides = DEC_CONV_STRIDES[5], activation=lrelu)(dec_h)
    dec_h = BatchNormalization()(dec_h)
    dec_h = Conv2DTranspose(filters = DEC_CONV_FILTERS[3], kernel_size = DEC_CONV_KERNEL_SIZES[1], strides = DEC_CONV_STRIDES[6], activation=lrelu)(dec_h)
    dec_h = BatchNormalization()(dec_h)

    dec_x = dec_h

    decoder = Model(dec_z, dec_x, name="decoder")
    decoder.summary()



    # --------- FULL VAE ---------
    vae = Model(enc_x, decoder(encoder(enc_x)), name="vae")


    # --------- TRAINING DEFINITIONS ---------

    def kl_loss(y_true, y_pred):
        y_true_flat = K.flatten(y_true)
        y_pred_flat = K.flatten(y_pred)
        kl = 0.5 * K.mean(K.square(enc_mn) + K.exp(enc_sd) - 1. - enc_sd)
        return kl

    def recon_loss(y_true, y_pred):
        y_true_flat = K.flatten(y_true)
        y_pred_flat = K.flatten(y_pred)
        recon = K.mean(K.square(y_true_flat - y_pred_flat))
        return recon

    beta = K.variable(value=beta_parameter)
    def vae_loss(y_true, y_pred):
        recon = recon_loss(y_true, y_pred)
        kl = kl_loss(y_true, y_pred)
        return recon + beta*kl


    Adam = optimizers.Adam(lr=learning_rate)
    vae.compile(optimizer=Adam, loss=vae_loss, metrics=[recon_loss, kl_loss])

    return vae, encoder, decoder, beta
