import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils
import math
import model
from parameters import *

tf.reset_default_graph()
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

raw_dataset = np.load("VG_data.npy")
dataset_size = len(raw_dataset)

if batch_size > dataset_size:
    batch_size = dataset_size

X_in = tf.placeholder(dtype=tf.float32, shape=[None, image_size[0], image_size[1], 3], name='X')
Y    = tf.placeholder(dtype=tf.float32, shape=[None, image_size[0], image_size[1], 3], name='Y')
Y_flat = tf.reshape(Y, shape=[-1, image_size[0] * image_size[1] * 3])
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')


sampled, mn, sd = model.encoder(X_in, keep_prob)
dec = model.decoder(sampled, keep_prob)

unreshaped = tf.reshape(dec, [-1, image_size[0]*image_size[1]*3])
img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
loss = tf.reduce_mean(img_loss + latent_loss)
optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())


dataset = tf.data.Dataset.from_tensor_slices(raw_dataset)
dataset = dataset.batch(batch_size)
iterator = dataset.make_initializable_iterator()
next_batch = iterator.get_next()

for i in range(epochs):
        print("----- EPOCH #%s -----" % (i+1))
        sess.run(iterator.initializer)
        try:
            iteration_count = 0
            n_iterations = math.ceil(dataset_size / batch_size)
            while True:
                    batch = sess.run(next_batch)
                    iteration_count += 1
                    (_, current_loss) = sess.run([optimizer, loss], feed_dict = {X_in: batch, Y: batch, keep_prob: 0.8})
                    print("iteration %d/%d - loss: %f" % (iteration_count, n_iterations, current_loss))
        except tf.errors.OutOfRangeError:
            pass

print("----- FINISHED TRAINING -----")

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
