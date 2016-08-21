import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, color, exposure
import numpy as np
import math


def convert_image(image, ppc, oris):
    """
    convert image to proper format (where the color channels are combined)

    args:
        image: np array X^2 x RGB
    """
    new = np.ones((image.shape[0], 1))
    for i in range(image.shape[0]):

        rgb = image[i][2]
        rgb = (rgb << 8) + image[i][1]
        rgb = (rgb << 8) + image[i][0]
        new[i] = rgb
    new = new.reshape(math.sqrt(image.shape[0]), math.sqrt(image.shape[0]))
    new = np.fliplr(new)
    new = np.rot90(new)
    data = color.rgb2gray(new)
    data = hog(new, orientations = oris, pixels_per_cell=(ppc, ppc),
        cells_per_block=(1, 1))
    return data


def print_hog_image(image):
    """
    image is expected to be in it's original format

    function prints hog image
    """
    print image.shape
    image = color.rgb2gray(image)

    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(4, 4),
                        cells_per_block=(1, 1), visualise=True, normalise=True)
    print "finished hog..."
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.show()

if __name__ == '__main__':
    image = color.rgb2gray(data.astronaut())
    print_hog_image(image)
