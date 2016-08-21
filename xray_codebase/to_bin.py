import math
import numpy as np
from scipy import misc
import struct

NUM_IMAGES = 300000
TEST = True

CLASS2VAL = {
    'airplane': 0,
    'automobile': 1,
    'bird': 2,
    'cat': 3,
    'deer': 4,
    'dog': 5,
    'frog': 6,
    'horse': 7,
    'ship': 8,
    'truck': 9
}

def write_test_data(num):
    """
    Write pngs to test data format:
    4 bytes: index
    3072 bytes: [r: 32x32 in row major form b:...  g:...]    
    """
    img_array = [misc.imread('PATH/TO/IMAGES/' + str(j + 1) + '.png')
                 for j in range(num)]
    with open("test.bin", 'w') as f:
        for j in range(num):  # length of your filename list
            if (j % 100 == 0): 
                print j
            image = img_array[j]
            # Image.show(Image.fromarray(np.asarray(image)))
            # write image index
            int32 = convert_to_bytearray(j)
            f.write(int32)
            write_image_slave(image, f)
            


def convert(num):    
    global TEST
    global VAL_SPLIT

    if TEST:
        write_test_data(num)
    else:
        write_train_data(num)

def write_train_data(num):
    """
    Write the pngs to training data forma:
    1 byte: label
    3072 bytes: [r:32x32 in row major form, b:same , g: same]
    """
    img_array = [misc.imread('data/train_images/' + str(j + 1) + '.png')
                 for j in range(num)]
    labels = load_labels(num)
    """
    Note: Having sklearn installed on our amazon instance would cause tensorflow
    to SegFault. For this reason, we used the prepackaged bins downloaded by 
    tensorflow to split our data in 80/20 splits. The following code is not used.    
    """
    X_train, X_val, y_train, y_val = None #cross_validation.train_test_split(
        #img_array, labels, test_size=VAL_SPLIT, random_state=0)
    with open("train.bin", 'w') as f:
        for j in range(len(X_train)):
            if (j%100 ==0):
                print "train ", j
            f.write(chr(y_train[j]))
            image = X_train[j]
            write_image_slave(image, f)
    
    with open("val.bin", 'w') as f:
        for j in range(len(X_val)):
            if (j%1000 == 0):
                print "val ", j
            f.write(chr(y_val[j]))
            image = X_val[j]
            write_image_slave(image, f)


def write_image_slave(image, f):
    """
    write image in channel -> row major form
    """
    r = (image[:, :, 0]).flatten()
    g = (image[:, :, 1]).flatten()
    b = (image[:, :, 2]).flatten()
    rgb = np.hstack((np.hstack((r, g)), b))
    for i in rgb:
        f.write(chr(i))


def load_labels(m=100):
    """
    Loads the training labels into numpy array
    """
    with open('data/trainLabels.csv', 'r') as f:
        f.readline()
        return np.array([CLASS2VAL[line.strip().split(',')[1]] for line in f])[:m]

def convert_to_bytearray(j):
    return struct.pack('>I', j)

if __name__ == "__main__":
    global NUM_IMAGES
    print NUM_IMAGES
    convert(NUM_IMAGES)
