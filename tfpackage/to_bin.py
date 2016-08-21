import math
import numpy as np
from scipy import misc
import struct



def convert(num):
    img_array = [misc.imread('test_images/' + str(j + 1) + '.png')
                 for j in range(num)]
    with open("test.bin", 'w') as f:
        for j in range(num):  # length of your filename list
            if (j % 100 == 0): 
                print j
            image = img_array[j]
            # Image.show(Image.fromarray(np.asarray(image)))
            # write image index
            three_bytes = convert_to_bytearray(j)
            f.write(three_bytes)
            r = (image[:, :, 0]).flatten()
            g = (image[:, :, 1]).flatten()
            b = (image[:, :, 2]).flatten()
            rgb = np.hstack((np.hstack((r, g)), b))
            for i in rgb:
                f.write(chr(i))


def convert_to_bytearray(j):
    return struct.pack('>I', j)

if __name__ == "__main__":
    convert(300000)
