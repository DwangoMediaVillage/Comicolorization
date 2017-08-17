import numpy
from PIL import Image
import subprocess
import sys
import tempfile


def detect_panel_rectangle(image_path, mfe_path, path_output=None):
    """
    Auto panel rectangle detection.
    https://github.com/DwangoMediaVillage/manga_frame_extraction
    :param image_path: page image path
    :param mfe_path: manga-frame-extraction's binary file path
    :param path_output: output directory path for panel images. if None, output to temporary directory
    :return: panel's rectangle [[left,top,width,height], ...]
    """
    rect_list = []
    if path_output is None:
        tmp = tempfile.TemporaryDirectory()
        path_output = tmp.name

    ret = subprocess.check_output([mfe_path, '-s', '-f', image_path, '-o', path_output]).decode(sys.stdout.encoding)

    w_scale = 1.0
    h_scale = 1.0
    for line in ret.split('\n'):
        words = line.split(' ')

        if words[0] == 'rescale':
            w_scale = float(words[1])
            h_scale = float(words[2])

        if words[0] == 'panel-region':
            rect = [int(num) for num in words[2:6]]
            rect[0] /= w_scale
            rect[1] /= h_scale
            rect[2] /= w_scale
            rect[3] /= h_scale
            rect_list.append(list(map(int, rect)))

    return rect_list


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
