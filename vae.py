import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import utils
import argparse
import math
import model
import gc
from parameters import *


parser = argparse.ArgumentParser()
parser.add_argument('experiment', help='name of the current experiment')
parser.add_argument('-s', '--save', help='save model weights flag', action="store_true")
args = parser.parse_args()
experiment = args.experiment
save_flag = args.save

dataset = np.load("fashion_data.npy")
dataset_size = len(dataset)

vae, encoder, decoder = model.build()

earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.00001, patience=5, verbose=1, mode='auto')
callbacks_list = [earlystop]
# callbacks_list = []

vae.fit(
    dataset, dataset,
    shuffle=True,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=validation_split,
    callbacks=callbacks_list
)

try:
    os.mkdir("experiments")
except:
    pass
model_filename = "experiments/{}_{}_{}_{}".format(experiment, n_latent, batch_size, epochs)
# if save_flag:
#     vae.save(model_filename+"_vae.h5")
#     del vae
#     gc.collect()
#     encoder.save(model_filename+"_enc.h5")
#     del encoder
#     gc.collect()
#     decoder.save(model_filename+"_dec.h5")
#     del decoder
#     gc.collect()

randoms = [np.random.normal(0, 1, n_latent) for _ in range(15)]
randoms = np.array(randoms)
imgs = decoder.predict(randoms, verbose=1)
imgs = [np.reshape(imgs[i], [image_size[0], image_size[1], 3]) for i in range(len(imgs))]
for img in imgs:
    plt.figure(figsize=(1,1))
    plt.axis('off')
    plt.imshow(img)
    plt.show()

np.random.shuffle(dataset)
latent_output = encoder.predict(dataset[:37])
dataset_output = decoder.predict(latent_output)
utils.saveComparisonImage(dataset[:37], dataset_output, model_filename)
