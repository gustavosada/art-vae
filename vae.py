import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
from parameters import *
import os, argparse, math, gc, time # python modules
import model, utils # project modules
K = tf.keras.backend

# get arguments
parser = argparse.ArgumentParser()
parser.add_argument('experiment', help='name of the current experiment')
parser.add_argument('-l', '--load', help='load checkpoint [filename]')
parser.add_argument('-s', '--save', help='save model weights flag', action="store_true")
args = parser.parse_args()
experiment = args.experiment
load = args.load
save_flag = args.save


# create experiment log directory
experiment_dir = utils.createExperimentDir(experiment)
if not experiment_dir:
    print("Already have an experiment with this name...")
    exit()

# load dataset
dataset = np.load("big_fashion.npy")
# dataset = dataset[:1000]
dataset_size = len(dataset)


vae, encoder, decoder, beta = model.build()
if load:
    filename = load
    vae.load_weights(os.path.join(experiment_dir, filename+".h5"))
    experiment_dir = os.path.join(experiment_dir, time())
    utils.createDir(experiment_dir)


# save model info
utils.saveModelDescription(encoder, decoder, experiment_dir)

# set training
def warmup(epoch):
    threshold_epoch = 5
    if epoch > threshold_epoch:
        value = 0.001
        ref_epoch = epoch - threshold_epoch
        value = value + ref_epoch * 0.00001
        value = min(1, value)
    else:
        value = 0
    print("beta:", value)
    K.set_value(beta, value)

def saveCurrentSampleCB(epoch):
    frequency = 1
    if epoch % frequency == 0 and epoch != 0:
        utils.saveCurrentSample(encoder, decoder, dataset[0], os.path.join(experiment_dir, "sample-{}.jpg".format(epoch)))

saveFrequency = 20
saveCB = tf.keras.callbacks.ModelCheckpoint(os.path.join(experiment_dir, "checkpoint-{epoch:02d}-{loss:02f}.h5"), period=saveFrequency, monitor='loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto')
warmupCB = tf.keras.callbacks.LambdaCallback(on_epoch_begin=lambda epoch, log: warmup(epoch))
saveImageCB = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, log: saveCurrentSampleCB(epoch))
tensorboard = TensorBoard(log_dir="logs/{}".format(experiment+" "+time()))
earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.00001, patience=5, verbose=10, mode='auto')

callbacks_list = [tensorboard, saveImageCB, saveCB]

if save_flag:
    callbacks_list.append(saveCB)

vae.fit(
    dataset, dataset,
    shuffle=True,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=validation_split,
    callbacks=callbacks_list
)

# save trained model info
utils.saveComparisonImages(encoder, decoder, dataset, experiment_dir)
utils.saveInterpolationImages(encoder, decoder, dataset, experiment_dir)


if save_flag:
    del dataset
    gc.collect()
    vae.save_weights(os.path.join(experiment_dir, "vae.h5"))
