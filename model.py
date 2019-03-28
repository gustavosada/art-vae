import tensorflow as tf
from parameters import *

dec_in_channels = 1 #NGM SABE QUE MERDA ÉH ESTA
reshaped_dim = [-1, 7, 7, dec_in_channels] #VERIFICAR ESSES 7 AI TCHE
inputs_decoder = 18 * dec_in_channels / 2

def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))


def encoder(X_in, keep_prob):
    activation = lrelu
    with tf.variable_scope("encoder", reuse=None):
        X = tf.reshape(X_in, shape=[-1, image_size[0], image_size[1], 3])
        x = tf.layers.conv2d(X, filters=32, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=32, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=32, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        mn = tf.layers.dense(x, units=n_latent)
        sd       = 0.5 * tf.layers.dense(x, units=n_latent)
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent]))
        z  = mn + tf.multiply(epsilon, tf.exp(sd))
        return z, mn, sd

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
