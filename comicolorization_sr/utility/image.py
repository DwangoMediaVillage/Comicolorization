import math
import numpy
import os
import subprocess
import typing
from PIL import Image
from skimage.color import lab2rgb


def array_to_image(images_array):
    # type: (numpy.ndarray) -> typing.List[Image.Image]
    color_images_array = images_array.transpose(0, 2, 3, 1)

    # to uint8
    minmax = (-1, 1)

    def clip_image(x):
        x = (x - minmax[0]) / (minmax[1] - minmax[0]) * 255  # normalize to 0~255
        return numpy.float32(0 if x < 0 else (255 if x > 255 else x))

    rgb_images_array = numpy.vectorize(clip_image)(color_images_array)
    rgb_images_array = rgb_images_array.astype(numpy.uint8)
    return [Image.fromarray(image_array) for image_array in rgb_images_array]


def lab_array_to_image(images_array, normalized=True):
    # type: (numpy.ndarray, any) -> typing.List[Image.Image]
    images_array = images_array.transpose(0, 2, 3, 1)

    if normalized:
        images_array[:, :, :, 0] = images_array[:, :, :, 0] + 1
        images_array *= 50

    def lab2image(image_array):
        image_array = image_array.astype(dtype=numpy.float64)
        rgb = (lab2rgb(image_array) * 255).astype(numpy.uint8)
        image = Image.fromarray(rgb)
        return image

    images = [lab2image(image_array) for image_array in images_array]
    return images


def save_images(images, path_directory, prefix_filename):
    """
    save image as [prefix_filename][index of image].png
    """
    # type: (typing.List[Image.Image], any, any) -> any
    if not os.path.exists(path_directory):
        os.mkdir(path_directory)

    filepath_list = []
    for i, image in enumerate(images):
        filename = prefix_filename + str(i) + '.png'
        filepath = os.path.join(path_directory, filename)
        image.save(filepath)
        filepath_list += [filepath]

    return filepath_list


def save_tiled_image(paths_input, path_output=None, col=None, row=None, border=5):
    # type: (typing.List[str], any, any, any, any) -> any
    num_image = len(paths_input)

    if path_output is None:
        commonpath = os.path.commonprefix(paths_input)
        path_output = commonpath + 'tiled.png'

    if col is None:
        col = math.ceil(math.sqrt(num_image))
    else:
        assert isinstance(col, int)

    if row is None:
        row = math.ceil(num_image / col)
    else:
        assert isinstance(row, int)

    assert isinstance(border, int)

    command = [
        'montage',
        '-tile', '{col}x{row}'.format(col=col, row=row),
        '-geometry', '+0',
        '-border' '{border}x{border}'.format(border=border),
        '{paths_input}'.format(paths_input=' '.join(paths_input)),
        '{path_output}'.format(path_output=path_output),
    ]
    subprocess.check_output(command)
