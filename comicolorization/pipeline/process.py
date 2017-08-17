import numpy
from PIL import Image


def make_binarized_image(base_image, threshold):
    """
    make binarized image
    :param base_image: source image
    :param threshold: threshold
    :return: binarized image
    """
    line = (numpy.array(base_image) > threshold).astype(numpy.uint8) * 255
    line = Image.fromarray(line)
    return line
