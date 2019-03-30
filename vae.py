import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils
import math
import model
from parameters import *

# CRIAR UM MAIN E DEFINIR ESSE ARQUIVO DE VAE COMO UMA CLASSE, ASSIM DA PRA USAR O TREINO
# DE DIVERSAS FORMAS, COMPARAR DIFERENTES PARAMETROS DE ENTRADA
# MAS ANTES AJEITAR A PARTE DE INPUT PARA PODER USAR OUTROS DATASETS COMO O DE MODA


to_categorical = tf.keras.utils.to_categorical
losses = tf.keras.losses
optimizers = tf.keras.optimizers

K = tf.keras.backend
K.set_epsilon(1e-05)

dataset = np.load("fashion_data.npy")
dataset_size = len(dataset)


sampled, mn, sd = model.encoder(X_in, keep_prob)
dec = model.decoder(sampled, keep_prob)

unreshaped = tf.reshape(dec, [-1, image_size[0]*image_size[1]*3])
img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
loss = tf.reduce_mean(img_loss + latent_loss)
optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print("PAUSA 3")
input()

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
