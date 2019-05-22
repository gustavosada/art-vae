import sys
import numpy as np
import os
import cv2
from parameters import *
from PIL import Image


def createExperimentDir(experiment_name):
    experiment_dir = os.path.join("experiments/", experiment_name)

    try:
        os.mkdir("experiments")
    except:
        pass

    try:
        os.mkdir(experiment_dir)
    except:
        # return False
        pass

    return experiment_dir

def createDir(dirpath):

    try:
        os.mkdir(dirpath)
    except:
        pass

    return dirpath



def saveModelDescription(encoder, decoder, experiment_dir):
    with open(os.path.join(experiment_dir, 'experiment.txt'), 'w') as file:
        import parameters
        for parameter_name in dir(parameters):
            if not parameter_name.startswith("__"):
                parameter_value = globals()[parameter_name]
                file.write("{}: {}\n".format(parameter_name, parameter_value))
        file.write("\n\n")
        encoder.summary(print_fn=lambda x: file.write(x + '\n'))
        file.write("\n\n")
        decoder.summary(print_fn=lambda x: file.write(x + '\n'))
    return True

def saveImage(image, filename):
    imagergb = Image.fromarray(ToRGB(np.uint8(255*image)))
    imagergb.save(filename)

def generateRandomImages(decoder):
    randoms = [np.random.normal(0, 0.2, n_latent) for _ in range(1)]
    randoms = np.array(randoms)
    imgs = decoder.predict(randoms, verbose=1)
    imgs = [np.reshape(imgs[i], [image_size[0], image_size[1], image_size[2]]) for i in range(len(imgs))]
    for img in imgs:
        plt.figure(figsize=(1,1))
        plt.axis('off')
        plt.imshow(img)
        plt.show()


def createInterpolationArray(latent1, latent2, n):
    out = []
    factor = 1/(n+1)
    for i in range(n):
        x = (i+1) * factor
        new_latent = x*latent2 + (1-x)*latent1 # latent2 in front because array appends from behind
        out.append(new_latent)
    return np.array(out)



def saveInterpolationImage(image1, i_array, image2, filename="output"):
    image1 = ToRGB(np.uint8(255*image1))
    i_array = ToRGB(np.uint8(255*i_array))
    image2 = ToRGB(np.uint8(255*image2))
    n = np.shape(i_array)[0]

    imY = np.shape(image1)[0]
    imX = np.shape(image1)[1]

    height = imY
    width = (2+n)*imX

    newImage = Image.new('RGB', (width, height))
    newImage.paste(Image.fromarray(image1), (0, 0))
    i = 1
    while i <= n:
        newImage.paste(Image.fromarray(i_array[i-1]), (i*imX, 0))
        i += 1
    newImage.paste(Image.fromarray(image2), (i*imX, 0))
    newImage.save(filename+".jpg")

def ToRGB(images):
    output = images
    if np.shape(images)[-1] == 1:
        output = np.reshape(output, np.shape(images)[:-1])
    return output


def saveInterpolationImages(encoder, decoder, dataset, experiment_dir):
    dirname = os.path.join(experiment_dir, "interpolations")

    try:
        os.mkdir(dirname)
    except:
        pass

    n = 10
    dataset_size = len(dataset)

    np.random.shuffle(dataset)
    latent_output = encoder.predict(dataset[:n])
    dataset_output = decoder.predict(latent_output)

    for i in range(n):
        for j in range(i, n):
            if i != j:
                interpolation_latent = createInterpolationArray(latent_output[i], latent_output[j], n)
                interpolation = decoder.predict(interpolation_latent)
                saveInterpolationImage(dataset_output[i], interpolation, dataset_output[j], os.path.join(dirname, "image{}{}".format(i, j)))



def saveComparisonImages(encoder, decoder, dataset, experiment_dir):
    n = 37
    filename = os.path.join(experiment_dir, "reconstruction")

    np.random.shuffle(dataset)
    latent_output = encoder.predict(dataset[:n])
    dataset_output = decoder.predict(latent_output)
    original = ToRGB(np.uint8(255*dataset[:n]))
    decoded = ToRGB(np.uint8(255*dataset_output))

    imY = np.shape(original)[1]
    imX = np.shape(original)[2]
    height = 8*imX
    width = 8*imY

    newImage = Image.new('RGB', (width, height))

    line = 0
    x_offset = 0

    for i in range(len(original)):
        if i % 4 == 0 and i != 0:
            line += 1
            x_offset = 0
        newImage.paste(Image.fromarray(original[i]), (x_offset,line*imY))
        newImage.paste(Image.fromarray(decoded[i]), (x_offset+imX,line*imY))
        x_offset += 2*imX

    newImage.save(filename + ".jpg")

def saveCurrentSample(encoder, decoder, data, filename):
    latent = encoder.predict(np.expand_dims(data, axis=0))
    y = decoder.predict(latent)
    saveImage(y[0], filename)
