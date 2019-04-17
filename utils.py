import sys
import numpy as np
from PIL import Image

# images = map(Image.open, ['Test1.jpg', 'Test2.jpg', 'Test3.jpg'])
# widths, heights = zip(*(i.size for i in images))
#
# total_width = sum(widths)
# max_height = max(heights)
#
# new_im = Image.new('RGB', (total_width, max_height))
#
# x_offset = 0
# for im in images:
#   new_im.paste(im, (x_offset,0))
#   x_offset += im.size[0]
#
# new_im.save('test.jpg')

#coded for van gogh dataset (37 images)
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
