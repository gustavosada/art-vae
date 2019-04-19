import sys
import numpy as np
from PIL import Image

# def saveInterpolationImage(image1, interpolation, image2, filename="output"):
#     image1 = np.uint8(255*image1)
#     interpolation = np.uint8(255*interpolation)
#     image2 = np.uint8(255*image2)
#
#     imY = np.shape(image1)[0]
#     imX = np.shape(image1)[1]
#
#     height = imY
#     width = 3*imX
#
#     newImage = Image.new('RGB', (width, height))
#     newImage.paste(Image.fromarray(image1), (0, 0))
#     newImage.paste(Image.fromarray(interpolation), (imX, 0))
#     newImage.paste(Image.fromarray(image2), (2*imX, 0))
#     newImage.save(filename+".jpg")

def saveInterpolationImage(image1, i_array, image2, filename="output"):
    image1 = np.uint8(255*image1)
    i_array = np.uint8(255*i_array)
    image2 = np.uint8(255*image2)
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

def saveComparisonImage(original, decoded, filename="output"):

    original = np.uint8(255*original)
    decoded = np.uint8(255*decoded)

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
