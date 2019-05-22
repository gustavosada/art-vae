import tensorflow as tf
from parameters import *

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

ENC_CONV_FILTERS =      [16, 32, 64, 128, 128]
ENC_CONV_KERNEL_SIZES = [3, 3, 3, 3, 3]
ENC_CONV_STRIDES =      [1, 1, 1, 1, 1]

DEC_DENSE = [2*2*128, image_size[0]*image_size[1]*image_size[2]]
DEC_RESHAPE = [2, 2, 128]

DEC_CONV_FILTERS = [128, 128, 64, 32, 16, image_size[2]]
DEC_CONV_KERNEL_SIZES = [3, 3, 3, 3, 3, 7]
DEC_CONV_STRIDES = [1, 1, 1, 1, 1, 1]


def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], n_latent), mean=0.,stddev=1)
    return z_mean + K.exp(z_log_var / 2) * epsilon

def build():


    # --------- ENCODER ---------
    enc_x = Input(shape=image_size)

    enc_h = Conv2D(filters = ENC_CONV_FILTERS[0], kernel_size = ENC_CONV_KERNEL_SIZES[0], strides = ENC_CONV_STRIDES[0], activation=lrelu)(enc_x)
    enc_h = BatchNormalization()(enc_h)
    enc_h = Conv2D(filters = ENC_CONV_FILTERS[0], kernel_size = ENC_CONV_KERNEL_SIZES[0], strides = ENC_CONV_STRIDES[0], activation=lrelu)(enc_h)
    enc_h = BatchNormalization()(enc_h)
    enc_h = MaxPooling2D((2, 2), padding='same')(enc_h)
    enc_h = Conv2D(filters = ENC_CONV_FILTERS[1], kernel_size = ENC_CONV_KERNEL_SIZES[1], strides = ENC_CONV_STRIDES[1], activation=lrelu)(enc_h)
    enc_h = BatchNormalization()(enc_h)
    enc_h = Conv2D(filters = ENC_CONV_FILTERS[1], kernel_size = ENC_CONV_KERNEL_SIZES[1], strides = ENC_CONV_STRIDES[1], activation=lrelu)(enc_h)
    enc_h = BatchNormalization()(enc_h)
    enc_h = MaxPooling2D((2, 2), padding='same')(enc_h)
    enc_h = Conv2D(filters = ENC_CONV_FILTERS[2], kernel_size = ENC_CONV_KERNEL_SIZES[2], strides = ENC_CONV_STRIDES[2], activation=lrelu)(enc_h)
    enc_h = BatchNormalization()(enc_h)
    enc_h = Conv2D(filters = ENC_CONV_FILTERS[2], kernel_size = ENC_CONV_KERNEL_SIZES[2], strides = ENC_CONV_STRIDES[2], activation=lrelu)(enc_h)
    enc_h = BatchNormalization()(enc_h)
    enc_h = Conv2D(filters = ENC_CONV_FILTERS[2], kernel_size = ENC_CONV_KERNEL_SIZES[2], strides = ENC_CONV_STRIDES[2], activation=lrelu)(enc_h)
    enc_h = BatchNormalization()(enc_h)
    enc_h = MaxPooling2D((2, 2), padding='same')(enc_h)
    enc_h = Conv2D(filters = ENC_CONV_FILTERS[3], kernel_size = ENC_CONV_KERNEL_SIZES[3], strides = ENC_CONV_STRIDES[3], activation=lrelu)(enc_h)
    enc_h = BatchNormalization()(enc_h)
    enc_h = Conv2D(filters = ENC_CONV_FILTERS[3], kernel_size = ENC_CONV_KERNEL_SIZES[3], strides = ENC_CONV_STRIDES[3], activation=lrelu)(enc_h)
    enc_h = BatchNormalization()(enc_h)
    enc_h = Conv2D(filters = ENC_CONV_FILTERS[3], kernel_size = ENC_CONV_KERNEL_SIZES[3], strides = ENC_CONV_STRIDES[3], activation=lrelu)(enc_h)
    enc_h = BatchNormalization()(enc_h)
    enc_h = MaxPooling2D((2, 2), padding='same')(enc_h)
    enc_h = Conv2D(filters = ENC_CONV_FILTERS[4], kernel_size = ENC_CONV_KERNEL_SIZES[4], strides = ENC_CONV_STRIDES[4], activation=lrelu)(enc_h)
    enc_h = BatchNormalization()(enc_h)
    enc_h = Conv2D(filters = ENC_CONV_FILTERS[4], kernel_size = ENC_CONV_KERNEL_SIZES[4], strides = ENC_CONV_STRIDES[4], activation=lrelu)(enc_h)
    enc_h = BatchNormalization()(enc_h)
    enc_h = Conv2D(filters = ENC_CONV_FILTERS[4], kernel_size = ENC_CONV_KERNEL_SIZES[4], strides = ENC_CONV_STRIDES[4], activation=lrelu)(enc_h)
    enc_h = BatchNormalization()(enc_h)
    enc_h = MaxPooling2D((2, 2), padding='same')(enc_h)

    enc_h = Flatten()(enc_h)
    # enc_h = Dense(4096)(enc_h)
    # enc_h = Dense(4096)(enc_h)
    # enc_h = Dense(1000)(enc_h)


    enc_mn = Dense(n_latent)(enc_h)
    enc_sd = Dense(n_latent)(enc_h)

    enc_z = Lambda(sampling)([enc_mn, enc_sd])

    encoder = Model(enc_x, enc_z, name="encoder")
    encoder.summary()


    # --------- DECODER ---------
    dec_z = Input(shape=(n_latent,))
    # dec_h = Dense(1000)(dec_z)
    # dec_h = Dense(4096)(dec_h)
    # dec_h = Dense(4096)(dec_h)

    dec_dense0 = Dense(DEC_DENSE[0])(dec_z)
    dec_r0 = Reshape(DEC_RESHAPE)(dec_dense0)
    dec_h = UpSampling2D((2, 2))(dec_r0)
    dec_h = Conv2DTranspose(filters = DEC_CONV_FILTERS[0], kernel_size = DEC_CONV_KERNEL_SIZES[0], strides = DEC_CONV_STRIDES[0], activation=lrelu)(dec_h)
    dec_h = BatchNormalization()(dec_h)
    dec_h = Conv2DTranspose(filters = DEC_CONV_FILTERS[0], kernel_size = DEC_CONV_KERNEL_SIZES[0], strides = DEC_CONV_STRIDES[0], activation=lrelu)(dec_h)
    dec_h = BatchNormalization()(dec_h)
    dec_h = UpSampling2D((2, 2))(dec_h)
    dec_h = Conv2DTranspose(filters = DEC_CONV_FILTERS[1], kernel_size = DEC_CONV_KERNEL_SIZES[1], strides = DEC_CONV_STRIDES[1], activation=lrelu)(dec_h)
    dec_h = BatchNormalization()(dec_h)
    dec_h = Conv2DTranspose(filters = DEC_CONV_FILTERS[1], kernel_size = DEC_CONV_KERNEL_SIZES[1], strides = DEC_CONV_STRIDES[1], activation=lrelu)(dec_h)
    dec_h = BatchNormalization()(dec_h)
    dec_h = Conv2DTranspose(filters = DEC_CONV_FILTERS[1], kernel_size = DEC_CONV_KERNEL_SIZES[1], strides = DEC_CONV_STRIDES[1], activation=lrelu)(dec_h)
    dec_h = BatchNormalization()(dec_h)
    dec_h = UpSampling2D((2, 2))(dec_h)
    dec_h = Conv2DTranspose(filters = DEC_CONV_FILTERS[2], kernel_size = DEC_CONV_KERNEL_SIZES[2], strides = DEC_CONV_STRIDES[2], activation=lrelu)(dec_h)
    dec_h = BatchNormalization()(dec_h)
    dec_h = Conv2DTranspose(filters = DEC_CONV_FILTERS[2], kernel_size = DEC_CONV_KERNEL_SIZES[2], strides = DEC_CONV_STRIDES[2], activation=lrelu)(dec_h)
    dec_h = BatchNormalization()(dec_h)
    dec_h = Conv2DTranspose(filters = DEC_CONV_FILTERS[2], kernel_size = DEC_CONV_KERNEL_SIZES[2], strides = DEC_CONV_STRIDES[2], activation=lrelu)(dec_h)
    dec_h = BatchNormalization()(dec_h)
    dec_h = UpSampling2D((2, 2))(dec_h)
    dec_h = Conv2DTranspose(filters = DEC_CONV_FILTERS[3], kernel_size = DEC_CONV_KERNEL_SIZES[3], strides = DEC_CONV_STRIDES[3], activation=lrelu)(dec_h)
    dec_h = BatchNormalization()(dec_h)
    dec_h = Conv2DTranspose(filters = DEC_CONV_FILTERS[3], kernel_size = DEC_CONV_KERNEL_SIZES[3], strides = DEC_CONV_STRIDES[3], activation=lrelu)(dec_h)
    dec_h = BatchNormalization()(dec_h)
    dec_h = Conv2DTranspose(filters = DEC_CONV_FILTERS[3], kernel_size = DEC_CONV_KERNEL_SIZES[3], strides = DEC_CONV_STRIDES[3], activation=lrelu)(dec_h)
    dec_h = BatchNormalization()(dec_h)
    dec_h = UpSampling2D((2, 2))(dec_h)
    dec_h = Conv2DTranspose(filters = DEC_CONV_FILTERS[4], kernel_size = DEC_CONV_KERNEL_SIZES[4], strides = DEC_CONV_STRIDES[4], activation=lrelu)(dec_h)
    dec_h = BatchNormalization()(dec_h)
    dec_h = Conv2DTranspose(filters = DEC_CONV_FILTERS[4], kernel_size = DEC_CONV_KERNEL_SIZES[4], strides = DEC_CONV_STRIDES[4], activation=lrelu)(dec_h)
    dec_h = BatchNormalization()(dec_h)
    dec_h = Conv2DTranspose(filters = DEC_CONV_FILTERS[4], kernel_size = DEC_CONV_KERNEL_SIZES[4], strides = DEC_CONV_STRIDES[4], activation=lrelu)(dec_h)
    dec_h = BatchNormalization()(dec_h)
    dec_h = Conv2DTranspose(filters = DEC_CONV_FILTERS[5], kernel_size = DEC_CONV_KERNEL_SIZES[5], strides = DEC_CONV_STRIDES[5], activation=lrelu)(dec_h)
    dec_h = BatchNormalization()(dec_h)

    # dec_f = Flatten()(dec_h)
    # dec_dense1 = Dense(DEC_DENSE[1], activation='sigmoid')(dec_f)
    # dec_x = Reshape(image_size)(dec_dense1)
    dec_x = dec_h

    decoder = Model(dec_z, dec_x, name="decoder")
    decoder.summary()



    # --------- FULL VAE ---------
    vae = Model(enc_x, decoder(encoder(enc_x)), name="vae")


    # --------- TRAINING DEFINITIONS ---------

    beta = K.variable(value=beta_parameter)
    def vae_loss(y_true, y_pred):
        y_true_flat = K.flatten(y_true)
        y_pred_flat = K.flatten(y_pred)

        recon = K.mean(K.square(y_true_flat - y_pred_flat))
        # recon = losses.categorical_crossentropy(y_true_flat, y_pred_flat)
        recon = K.print_tensor(recon, message='\nrecon = ')
        # # recon = K.sum(losses.binary_crossentropy(y_true_flat, y_pred_flat), axis=1)
        kl = 0.5 * K.mean(K.square(enc_mn) + K.exp(enc_sd) - 1. - enc_sd)
        kl = K.print_tensor(kl, message='\nkl = ')
        # c = K.print_tensor(np.shape(kl), message='\nkl = ')
        # # kl = 0
        return recon + beta*kl
        # return recon

    # def vae_loss(y_true, y_pred):
    #     vae_z_log_var = enc_sd
    #     vae_z_mean = enc_mn
    #     y_pred = K.clip(y_pred,1e-05,50)
    #     recon = K.sum(losses.categorical_crossentropy(y_true, y_pred))
    #     recon = K.print_tensor(recon, message='\nrecon = ')
    #     kl = - 0.5 * beta_parameter * K.sum(1 + vae_z_log_var - K.square(vae_z_mean) - K.exp(vae_z_log_var), axis=-1)
    #     kl = K.print_tensor(kl, message='\nkl = ')
    #     return K.mean(recon + kl)

    # def vae_loss(x, x_decoded_mean):
    # xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
    # kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    # return xent_loss + kl_loss

    Adam = optimizers.Adam(lr=learning_rate)
    vae.compile(optimizer=Adam, loss=vae_loss)

    return vae, encoder, decoder, beta
