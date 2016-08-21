import numpy as np
import hogify
from PIL import Image
from multiprocessing import Pool

ORIS = 8
SIZE = 32
NUM_PROCESSES = 12

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
VAL2CLASS = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}


def load_hog_slave(args):
    ppc, index, train = args
    pix = Image.open('data/' + ('train/' if train else 'test/') + str(index + 1) + '.png').load()
    return hogify.convert_image(np.array([pix[x, y] for x in xrange(SIZE) for y in xrange(SIZE)]), ppc, ORIS)


def load_hog(ppc, m=100, train=True):
    global X
    factor = 1.0 * ORIS / (ppc * ppc)
    # X = np.ones((m, 200))
    p = Pool(processes=NUM_PROCESSES)
    X = np.array(p.map(load_hog_slave, [(ppc, i, train) for i in xrange(m)]))
    return X, load_labels(m) if train else None


def load_images_slave(args):
    index, train = args
    pix = Image.open('data/' + ('train/' if train else 'test/') + str(index + 1) + '.png').load()
    X[index, 1:] = np.array([pix[x, y] for x in xrange(SIZE) for y in xrange(SIZE)]).reshape(1, X.shape[1])

def load_image_vanilla(index):
    pix = Image.open('data/' + 'test/test/' + str(index + 1) + '.png').load()
    X_test[index,:] = np.array([pix[x, y] for x in xrange(SIZE) for y in xrange(SIZE)]).reshape(1, X_test.shape[1])


def load_images(m=100, train=True):
    """
    Loads training data into numpy array.
    Each 32x32 pixel image is divided into RGB values for 32x32x3 features (3072).
    A column of 1s is added for the intercept (3073).
    The mean image is subtracted.
    Note that the filenames are 1 indexed (there is no 0.png).
    m -- number of examples to use
    Returns the X data and the y label data
    """
    global X
    d = 3 * SIZE * SIZE
    X = np.ones((m, d + 1))  # 3073 features, intercept + 32x32xRGB
    p = Pool(processes=NUM_PROCESSES)
    p.map(load_images_slave, [(i, train) for i in xrange(m)])

    # subtract the mean image from X
    X[:, 1:] -= np.mean(X[:, 1:], axis=0)
    return X, load_labels(m) if train else None


def load_labels(m=100):
    """
    Loads the training labels into numpy array
    """
    with open('data/trainLabels.csv', 'r') as f:
        f.readline()
        return np.array([CLASS2VAL[line.strip().split(',')[1]] for line in f])[:m]

def convert_jpg_to_bin(m=300000):
    """
    Convert the test jpgs into bin images and write output
    """
    global X_test    
    d = 3 * SIZE * SIZE
    X_test = np.ones((m, d))
    for i in xrange(m):
        load_image_vanilla(i)
    return write_to_file(X_test)

def write_to_file(data):
    "Write matrix to bin file"

    n = open('test_images.bin', 'w')
    f = open('numbers', 'w')
    for num in data:
        n.write(num)
        f.write(str(num))   

def format_output(y, filename='results.csv'):
    """
    Takes a list of labels and outputs in the Kaggle required format.
    Note that the y array will be zero indexed, but Kaggle data must be 1 indexed.
    y -- [1, 3, 5, 2, 4, 0]
    output file: "id,label\n1,automobile\n2,cat\n...
    """
    with open(filename, 'wb') as f:
        f.write('id,label\n')
        for i, val in enumerate(y):
            f.write(str(i + 1) + ',' + VAL2CLASS[val] + '\n')    