import tensorflow as tf
import numpy as np
import model
import utils

model_filename = 'experiments/2conv_70_32_100'
n = 5

vae, encoder, decoder = model.build()

vae.load_weights(model_filename+"_vae.h5")
encoder.load_weights(model_filename+"_enc.h5")
decoder.load_weights(model_filename+"_dec.h5")

dataset = np.load("fashion_data.npy")
dataset_size = len(dataset)

# np.random.shuffle(dataset)
latent_output = encoder.predict(dataset[:n])
dataset_output = decoder.predict(latent_output)

def createInterpolationArray(latent1, latent2, n):
    out = []
    factor = 1/(n+1)
    for i in range(n):
        x = (i+1) * factor
        new_latent = x*latent2 + (1-x)*latent1 # latent2 in front because array appends from behind
        out.append(new_latent)
    return np.array(out)

for i in range(n):
    for j in range(i, n):
        if i != j:
            interpolation_latent = createInterpolationArray(latent_output[i], latent_output[j], 5)
            interpolation = decoder.predict(interpolation_latent)
            utils.saveInterpolationImage(dataset_output[i], interpolation, dataset_output[j], "images/image{}{}".format(i, j))
