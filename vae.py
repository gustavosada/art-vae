import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils

tf.reset_default_graph()
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

raw_dataset = np.load("VG_data.npy")
dataset_size = len(raw_dataset)

n_latent = 50
image_size = [256, 256]
epochs = 1000
batch_size = 5
if batch_size > dataset_size:
    batch_size = dataset_size

X_in = tf.placeholder(dtype=tf.float32, shape=[None, image_size[0], image_size[1], 3], name='X')
Y    = tf.placeholder(dtype=tf.float32, shape=[None, image_size[0], image_size[1], 3], name='Y')
Y_flat = tf.reshape(Y, shape=[-1, image_size[0] * image_size[1] * 3])
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

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

sampled, mn, sd = encoder(X_in, keep_prob)
dec = decoder(sampled, keep_prob)

unreshaped = tf.reshape(dec, [-1, image_size[0]*image_size[1]*3])
img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
loss = tf.reduce_mean(img_loss + latent_loss)
optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# dataset = tf.data.Dataset.from_tensor_slices(raw_dataset)
# # dataset.repeat(epochs).batch(batch_size)
# iterator = dataset.make_one_shot_iterator()

# data = []
#
# next = iterator.get_next()
# for i in range(dataset_size):
#     element = sess.run(next)
#     data.append(element)

data = raw_dataset

for i in range(epochs):
    sess.run(optimizer, feed_dict = {X_in: data, Y: data, keep_prob: 0.8})

randoms = [np.random.normal(0, 1, n_latent) for _ in range(5)]
imgs = sess.run(dec, feed_dict = {sampled: randoms, keep_prob: 1.0})
imgs = [np.reshape(imgs[i], [image_size[0], image_size[1], 3]) for i in range(len(imgs))]

for img in imgs:
    plt.figure(figsize=(1,1))
    plt.axis('off')
    plt.imshow(img)
    plt.show()

latent_output = sess.run(sampled, feed_dict = {X_in: raw_dataset, keep_prob: 1.0})
dataset_output = sess.run(dec, feed_dict = {sampled: latent_output, keep_prob: 1.0})
utils.saveComparisonImage(raw_dataset, dataset_output)


exit()
