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


dataset = np.load("fashion_data.npy")
dataset_size = len(dataset)

vae = model.build()

earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
callbacks_list = [earlystop]

vae.fit(
    dataset, dataset,
    shuffle=True,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=validation_split,
    callbacks=callbacks_list
)

# randoms = [np.random.normal(0, 1, n_latent) for _ in range(5)]
# imgs = sess.run(dec, feed_dict = {sampled: randoms, keep_prob: 1.0})
# imgs = [np.reshape(imgs[i], [image_size[0], image_size[1], 3]) for i in range(len(imgs))]
#
# for img in imgs:
#     plt.figure(figsize=(1,1))
#     plt.axis('off')
#     plt.imshow(img)
#     plt.show()
#
# latent_output = sess.run(sampled, feed_dict = {X_in: raw_dataset, keep_prob: 1.0})
# dataset_output = sess.run(dec, feed_dict = {sampled: latent_output, keep_prob: 1.0})
# utils.saveComparisonImage(raw_dataset, dataset_output)
#
#
# exit()
