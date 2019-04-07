import os
import argparse
from keras.preprocessing import image as image_utils
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt
import numpy as np

#----------- CONSTANTS ------------

IMAGE_SIZE = (100, 100)

#----------------------------------


#----------- PARSE PARAMETERS ------------
parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', help='path of the dataset')
parser.add_argument('--size', '-s', help='square dimension of the image (only one number)')
args = parser.parse_args()
dataPath = args.data
imageSizeArg = int(args.size)

if not dataPath:
    # print('error: specify dataset with --data arg')
    # exit()
    dataPath = 'VG_data'

IMAGE_SIZE = (imageSizeArg, imageSizeArg)
#------------------------------------------


dataset = []
imageList = os.listdir(dataPath)
print("Found {0} images in {1} dataset".format(len(imageList), dataPath))

idx = 1
for imageName in imageList:
    imagePath = os.path.join(dataPath, imageName)
    image = image_utils.load_img(imagePath).resize(IMAGE_SIZE, Image.ANTIALIAS)
    imageData = np.array(image.getdata()).reshape(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    imageData = imageData.astype('float32')/255
    # imageData = imageData.astype('float32')
    dataset.append(imageData)
    print(str(idx)+'- Loaded image '+imageName)
    idx+=1

dataset = np.array(dataset)
print('Dataset Shape:', dataset.shape)
np.save(dataPath, dataset)
print('Succesfully created {0}.npy'.format(dataPath))
