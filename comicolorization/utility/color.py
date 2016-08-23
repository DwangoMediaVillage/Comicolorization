import numpy
from skimage.color import rgb2lab


def _calc_rgb2lab_min_max():
    """
    :return: ([L_min, a_min, b_min], [L_max, a_max, b_max])
    """
    num_space = 16
    size_image = num_space * num_space * num_space
    values_pixel = numpy.linspace(0, 1, num_space)

    image_array = [[r, g, b] for r in values_pixel for g in values_pixel for b in values_pixel]
    image_array = numpy.vstack(image_array).reshape((1, size_image, 3))

    image_array = rgb2lab(image_array)  # illuminant='D65'
    return image_array.min(axis=1).squeeze(), image_array.max(axis=1).squeeze()


lab_min_max = _calc_rgb2lab_min_max()


def normalize(array, in_min, in_max, out_min, out_max):
    in_range = in_max - in_min
    out_range = out_max - out_min
    return (array - in_min) / in_range * out_range + out_min


def normalize_each_channel(array, in_min, in_max, out_min, out_max, split, concat):
    channels = tuple(
        normalize(channel, in_min[i], in_max[i], out_min[i], out_max[i])
        for i, channel in enumerate(split(array))
    )
    return concat(channels)


def normalize_zero_one(array, in_type, split, concat):
    out_min = (0, 0, 0)
    out_max = (1, 1, 1)

    if in_type == 'RGB':
        in_min = (0, 0, 0)
        in_max = (255, 255, 255)
    elif in_type == 'Lab':
        in_min, in_max = lab_min_max
    elif in_type == 'ab':
        in_min, in_max = lab_min_max
        in_min = in_min[1:]
        in_max = in_max[1:]
        out_min = out_min[1:]
        out_max = out_max[1:]
    else:
        raise ValueError(in_type)

    return normalize_each_channel(array, in_min, in_max, out_min, out_max, split, concat)
