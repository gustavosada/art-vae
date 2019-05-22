import os
import argparse
from keras.preprocessing import image as image_utils
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt
import numpy as np
import cv2

#----------- CONSTANTS ------------

IMAGE_SIZE = (100, 100)

#----------------------------------


#----------- PARSE PARAMETERS ------------
parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', help='path of the dataset')
parser.add_argument('--size', '-s', help='square dimension of the image (only one number)')
parser.add_argument('-g', '--grayscale', help='save images in grayscale', action="store_true")
args = parser.parse_args()
dataPath = args.data
imageSizeArg = int(args.size)
gray_flag = args.grayscale

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

    if gray_flag:
        ii = cv2.imread(imagePath)
        ii = cv2.resize(ii, IMAGE_SIZE)
        gray_image = cv2.cvtColor(ii, cv2.COLOR_BGR2GRAY)
        imageData = gray_image
        imageData = np.reshape(imageData, (IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
    else:
        image = image_utils.load_img(imagePath).resize(IMAGE_SIZE, Image.ANTIALIAS)
        imageData = np.array(image.getdata()).reshape(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)

    imageData = imageData.astype('float32')/255
    dataset.append(imageData)
    print(str(idx)+'- Loaded image '+imageName)
    idx+=1

dataset = np.array(dataset)
print('Dataset Shape:', dataset.shape)
np.save(dataPath, dataset)
print('Succesfully created {0}.npy'.format(dataPath))
