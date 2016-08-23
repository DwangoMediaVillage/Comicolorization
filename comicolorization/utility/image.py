import os
import chainer
import numpy
import typing
from PIL import Image
from skimage.color import lab2rgb


def array_to_image(
        color_images_array=None,
        gray_images_array=None,
        mode='RGB',
        color_normalize=False,
        linedrawing=None,
) -> typing.List[Image.Image]:
    """
    :param color_images_array: shape is [number of image, channel(3), width, height]
    :param gray_images_array: used when mode=='ab' or 'gray'
    :param mode: mode of input images array (RGB, Lab, ab, gray)
    :param color_normalize: normalize rgb color to [min(rgb_images_array) max(rgb_images_array)]
    """
    if color_images_array is not None:
        color_images_array = chainer.cuda.to_cpu(color_images_array)

    if gray_images_array is not None:
        gray_images_array = chainer.cuda.to_cpu(gray_images_array)

    if mode == 'gray':
        color_images_array = numpy.concatenate([gray_images_array] * 3, axis=1)
        mode = 'RGB'

    if mode == 'ab':
        # concat gray image(luminance) and ab(chromaticity)
        color_images_array = chainer.cuda.to_cpu(color_images_array)
        color_images_array = numpy.concatenate((gray_images_array, color_images_array), axis=1)
        mode = 'Lab'

    color_images_array = color_images_array.transpose(0, 2, 3, 1)

    if mode == 'Lab':
        color_images_array = color_images_array.astype(dtype=numpy.float64)
        image_array_list = [lab2rgb(image_array) * 255 for image_array in color_images_array]
        color_images_array = numpy.concatenate(
            [numpy.expand_dims(image_array, axis=0) for image_array in image_array_list]
        )
        mode = 'RGB'

    if mode == 'RGB':
        rgb_images_array = color_images_array
    else:
        raise ValueError('{} mode is not supported'.format(mode))

    # to uint8
    if color_normalize:
        minmax = (rgb_images_array.min(), rgb_images_array.max())
    else:
        if linedrawing is not None:
            minmax = (0, 1)
        else:
            minmax = (0, 255)

    def clip_image(x):
        x = (x - minmax[0]) / (minmax[1] - minmax[0]) * 255  # normalize to 0~255
        return numpy.float32(0 if x < 0 else (255 if x > 255 else x))

    rgb_images_array = numpy.vectorize(clip_image)(rgb_images_array)
    rgb_images_array = rgb_images_array.astype(numpy.uint8)
    return [Image.fromarray(image_array) for image_array in rgb_images_array]


def save_images(images: typing.List[Image.Image], path_directory, prefix_filename, index_base=0):
    """
    save image as [prefix_filename][index of image].png
    """
    if not os.path.exists(path_directory):
        os.mkdir(path_directory)

    for i, image in enumerate(images):
        filename = prefix_filename + str(index_base + i) + '.png'
        filepath = os.path.join(path_directory, filename)
        image.save(filepath)


def make_histogram(
        image_array,
        num_bins,
        multidim: bool,
        threshold_palette=None,
        ranges=((0, 255), (0, 255), (0, 255)),
):
    channel, x, y = image_array.shape
    if not multidim:
        histogram_one = []
        for h_channel, range in zip(image_array, ranges):
            hist = numpy.histogram(h_channel, num_bins, range=range)[0]
            histogram_one.append(hist)
    else:
        h_each_channel = numpy.reshape(image_array, (channel, x * y)).T
        bins_each_channel = numpy.asarray([num_bins] * channel)
        histogram_one = numpy.histogramdd(h_each_channel, bins_each_channel, range=ranges)[0]

    hist = numpy.asarray(histogram_one) / (x * y)
    if threshold_palette is not None:
        palette = numpy.zeros(shape=hist.shape)
        palette[hist > threshold_palette] = 1
        hist = palette

    hist = hist.reshape(-1)
    return hist.astype(image_array.dtype)


def rebalance_top_histogram(histogram, rate: float):
    # if rate > 1, collect to top bin
    # if rate < 1, distribute from top bin
    assert histogram.ndim == 1

    s = histogram.sum()
    top_index = histogram.argmax()

    top = histogram[top_index]
    top_after = top * rate
    if top_after > s:
        top_after = s

    other = s - top
    other_after = s - top_after

    output = histogram / other * other_after  # type: numpy.ndarray
    output[top_index] = top_after
    return output


def draw(
        model, input_images_array,
        rgb_images_array=None,
        histogram_image_array=None, histogram_array=None,
        gpu=-1,
):
    # histogram
    assert histogram_image_array is None or histogram_array is None

    if histogram_image_array is not None:
        histogram_image_array = histogram_image_array[numpy.newaxis, :, :, :]
        rgb_images_array = numpy.repeat(histogram_image_array, len(input_images_array), axis=0)

    if histogram_array is not None:
        rgb_images_array = None

    if gpu >= 0:
        input_images_array = chainer.cuda.to_gpu(input_images_array, gpu)

    return model.generate_rgb_image(
        input_images_array,
        rgb_images_array=rgb_images_array,
        histogram_array=histogram_array,
    )
