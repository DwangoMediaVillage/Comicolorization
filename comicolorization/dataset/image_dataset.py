from __future__ import division
from chainer.dataset import dataset_mixin
from abc import ABCMeta, abstractmethod
import os
from PIL import Image
from PIL import ImageOps
import numpy
import six
import skimage.filters
from skimage.color import rgb2lab
from skimage import exposure
import cv2
import time

from comicolorization.utility import color

"""
Just a little bit modification

@see https://github.com/pfnet/chainer/blob/master/chainer/datasets/image_dataset.py
"""


@six.add_metaclass(ABCMeta)
class InputOutputDatsetInterface(object):
    @abstractmethod
    def get_input_luminance_range(self):
        pass

    @abstractmethod
    def get_input_range(self):
        pass

    @abstractmethod
    def get_output_range(self):
        pass


class PILImageDatasetBase(dataset_mixin.DatasetMixin):
    def __init__(self, paths, resize=None, random_crop_size=None, random_flip=False, test=False, root='.'):
        """
        :param resize: if it is not None, resize image
        :param random_crop_size: if it is not None, random crop image after resize
        :param random_flip: if it is True, random flip image right left
        """
        if isinstance(paths, six.string_types):
            with open(paths) as paths_file:
                paths = [path.strip() for path in paths_file]
        self._paths = paths
        self._root = root
        self._resize = resize
        self._crop_size = random_crop_size
        self._flip = random_flip
        self._test = test

    def __len__(self):
        return len(self._paths)

    def get_example(self, i):
        # type: (any) -> (str, Image)
        path = os.path.join(self._root, self._paths[i])
        image = Image.open(path)

        if self._resize is not None:
            image = image.resize(self._resize)

        if self._crop_size is not None:
            width, height = image.size

            if self._test is True:
                top = int((height - self._crop_size[1]) / 2)
                left = int((width - self._crop_size[0]) / 2)
                bottom = top + self._crop_size[1]
                right = left + self._crop_size[0]

            else:
                top = numpy.random.randint(height - self._crop_size[1] + 1)
                left = numpy.random.randint(width - self._crop_size[0] + 1)
                bottom = top + self._crop_size[1]
                right = left + self._crop_size[0]

            image = image.crop((left, top, right, bottom))

        if self._flip:
            if numpy.random.randint(2) == 1:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

        return path, image


class PILImageDataset(PILImageDatasetBase):
    def get_example(self, i):
        # type: (any) -> Image
        return super(PILImageDataset,self).get_example(i)[1]


class ColorMonoImageDataset(dataset_mixin.DatasetMixin, InputOutputDatsetInterface):
    def __init__(self, base, dtype=numpy.float32):
        # type: (PILImageDataset, any) -> any
        self._dtype = dtype
        self.base = base

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        # type: (any) -> (numpy.ndarray, numpy.ndarray, numpy.ndarray)
        """
        :return: (RGB array [0~255], gray array [0~255], RGB array [0~255])
        """
        image = self.base[i]
        rgb_image_data = numpy.asarray(image, dtype=self._dtype).transpose(2, 0, 1)[:3, :, :]
        gray_image = ImageOps.grayscale(image)
        gray_image_data = numpy.asarray(gray_image, dtype=self._dtype)[:, :, numpy.newaxis].transpose(2, 0, 1)
        return rgb_image_data, gray_image_data, rgb_image_data

    def get_input_luminance_range(self):
        raise NotImplementedError

    def get_input_range(self):
        return (0, 255)

    def get_output_range(self):
        return (0, 255), (0, 255), (0, 255)


class LabImageDataset(dataset_mixin.DatasetMixin, InputOutputDatsetInterface):
    def __init__(self, base):
        # type: (ColorMonoImageDataset)-> None
        self.base = base

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        # type: (any) -> (numpy.ndarray, numpy.ndarray, numpy.ndarray)
        rgb_image_data, gray_image_data, _ = self.base[i]
        dtype = rgb_image_data.dtype
        image_data = rgb_image_data.transpose(1, 2, 0) / 255
        lab_image_data = rgb2lab(image_data).transpose(2, 0, 1).astype(dtype)
        luminous_image_data = numpy.expand_dims(lab_image_data[0], axis=0)
        return lab_image_data, luminous_image_data, rgb_image_data

    def get_input_luminance_range(self):
        return (color.lab_min_max[0][0], color.lab_min_max[1][0])

    def get_input_range(self):
        return (color.lab_min_max[0][0], color.lab_min_max[1][0])

    def get_output_range(self):
        return tuple((color.lab_min_max[0][i], color.lab_min_max[1][i]) for i in range(3))


class LabOnlyChromaticityDataset(dataset_mixin.DatasetMixin, InputOutputDatsetInterface):
    def __init__(self, base):
        # type: (LabImageDataset) -> None
        self.base = base

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        # type: (any) -> (numpy.ndarray, numpy.ndarray, numpy.ndarray)
        lab_image_data, luminous_image_data, rgb_image_data = self.base[i]
        return lab_image_data[1:], luminous_image_data, rgb_image_data

    def get_input_luminance_range(self):
        return self.base.get_input_luminance_range()

    def get_input_range(self):
        return self.base.get_input_range()

    def get_output_range(self):
        return self.get_output_range()[1:]


class LineDrawingDatasetBase(dataset_mixin.DatasetMixin, InputOutputDatsetInterface):
    def __init__(self, base):
        # type: (LabImageDataset) -> None
        self.base = base

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        # type: (any) -> (numpy.ndarray, numpy.ndarray, numpy.ndarray)
        lab_image_data, luminous_image_data, rgb_image_data = self.base[i]
        luminous_image_data = numpy.squeeze(luminous_image_data).astype(numpy.uint8)
        linedrawing = self.convert_to_linedrawing(luminous_image_data)
        linedrawing = linedrawing.astype(numpy.float32) / 255
        linedrawing = numpy.expand_dims(linedrawing, axis=0)
        return lab_image_data, linedrawing, rgb_image_data

    def convert_to_linedrawing(self, luminous_image_data):
        raise NotImplementedError()

    def get_input_luminance_range(self):
        return (0, 1)

    def get_input_range(self):
        return (0, 1)

    def get_output_range(self):
        return self.base.get_output_range()


class LabOtsuThresholdImageDataset(LineDrawingDatasetBase):
    def convert_to_linedrawing(self, luminous_image_data):
        linedrawing = cv2.threshold(luminous_image_data, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        return linedrawing


class LabAdaptiveThresholdImageDataset(LineDrawingDatasetBase):
    def convert_to_linedrawing(self, luminous_image_data):
        linedrawing = cv2.adaptiveThreshold(luminous_image_data, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                            11, 2)
        return linedrawing


class LabCannyImageDataset(LineDrawingDatasetBase):
    def convert_to_linedrawing(self, luminous_image_data):
        kernel = numpy.ones((3, 3), numpy.uint8)
        linedrawing = cv2.Canny(luminous_image_data, 5, 125)
        linedrawing = cv2.bitwise_not(linedrawing)
        linedrawing = cv2.erode(linedrawing, kernel, iterations=1)
        linedrawing = cv2.dilate(linedrawing, kernel, iterations=1)
        return linedrawing


class LabThreeValueThresholdImageDataset(LineDrawingDatasetBase):
    def convert_to_linedrawing(self, luminous_image_data):
        hist = exposure.histogram(luminous_image_data)[0]
        hist = hist.reshape(numpy.prod(hist.shape))
        n = len(hist)

        value = numpy.arange(n, dtype=numpy.float)

        # find best thresholds
        t1 = 0
        t2 = n
        max_var = 0
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                w1 = numpy.sum(hist[0:i])
                w2 = numpy.sum(hist[i:j])
                w3 = numpy.sum(hist[j:n])
                if w1 == 0 or w2 == 0 or w3 == 0:
                    continue
                m1 = numpy.dot(hist[0:i], value[0:i]) / w1
                m2 = numpy.dot(hist[i:j], value[i:j]) / w2
                m3 = numpy.dot(hist[j:n], value[j:n]) / w3
                var = w1 * w2 * (m1 - m2) ** 2 + \
                      w2 * w3 * (m2 - m3) ** 2 + \
                      w3 * w1 * (m3 - m1) ** 2
                if max_var < var:
                    max_var = var
                    t1 = i
                    t2 = j

        # quantize
        retimg = luminous_image_data.reshape(numpy.prod(luminous_image_data.shape))
        for i in range(retimg.size):
            value = retimg[i]
            if value < t1:
                value = 0
            elif value < t2:
                value = 127
            else:
                value = 255
            retimg[i] = value
        return retimg.reshape(luminous_image_data.shape)


class LabDilateDiffImageDataset(LineDrawingDatasetBase):
    def convert_to_linedrawing(self, luminous_image_data):
        neiborhood24 = numpy.array([[1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1]],
                                   numpy.uint8)
        dilated = cv2.dilate(luminous_image_data, neiborhood24, iterations=1)
        diff = cv2.absdiff(dilated, luminous_image_data)
        linedrawing = cv2.bitwise_not(diff)
        return linedrawing


class LabSeveralPixelDrawingImageDataset(dataset_mixin.DatasetMixin, InputOutputDatsetInterface):
    def __init__(
            self,
            base,
            max_point,
            max_size,
            fix_position=False,
    ):
        # type: (LineDrawingDatasetBase, int, int, any) -> None
        """
        :param max_point: max number of drawing point (is not pixel size)
        """
        self.base = base
        self.max_point = max_point
        self.max_size = max_size

        self.fix_position = fix_position

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        # type: (any) -> (numpy.ndarray, numpy.ndarray, numpy.ndarray)
        lab_image_data, linedrawing, rgb_image_data = self.base[i]
        color_linedrawing = numpy.pad(linedrawing, ((0, 2), (0, 0), (0, 0)), mode='constant', constant_values=0)

        width, height = linedrawing.shape[1:]

        if not self.fix_position:
            num_point = numpy.random.randint(low=0, high=self.max_point)
            size_point = numpy.random.randint(low=0, high=self.max_size) + 1

            top_points = numpy.asarray(
                [[x, y] for x in range(width - (size_point - 1)) for y in range(height - (size_point - 1))])
            top_points = numpy.random.permutation(top_points)[:num_point]

            expanded_points_list = [top_points + numpy.array([x, y]) for x in range(size_point) for y in
                                    range(size_point)]
            points = numpy.concatenate(expanded_points_list)
        else:
            nx = ny = 3
            xs = numpy.linspace(0, width - 1, nx + 2, dtype=numpy.int32)[1:-1]
            ys = numpy.linspace(0, width - 1, ny + 2, dtype=numpy.int32)[1:-1]
            points = numpy.asarray([[x, y] for x in xs for y in ys])

        for x, y in points:
            linedrawing_luminance = self.get_input_luminance_range()
            lab_luminance = self.get_output_range()[0]
            color_linedrawing[0, x, y] = color.normalize(
                lab_image_data[0, x, y],
                in_min=lab_luminance[0], in_max=lab_luminance[1],
                out_min=linedrawing_luminance[0], out_max=linedrawing_luminance[1],
            )
            color_linedrawing[1:, x, y] = lab_image_data[1:, x, y]

        return lab_image_data, color_linedrawing, rgb_image_data

    def get_input_luminance_range(self):
        return self.base.get_input_luminance_range()

    def get_input_range(self):
        output_range = self.base.get_output_range()
        return self.base.get_input_luminance_range(), output_range[1], output_range[2]

    def get_output_range(self):
        return self.base.get_output_range()


class BinarizationImageDataset(dataset_mixin.DatasetMixin):
    def __init__(self, base):
        # type: (LabImageDataset) -> None
        self.base = base

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        # type: (any) -> (numpy.ndarray, numpy.ndarray, numpy.ndarray)
        lab_image_data, luminous_image_data, rgb_image_data = self.base[i]

        threshold = skimage.filters.threshold_otsu(luminous_image_data)

        binarized_image = numpy.zeros(luminous_image_data.shape, dtype=luminous_image_data.dtype)
        binarized_image[luminous_image_data > threshold] = 1
        return lab_image_data, binarized_image, rgb_image_data

    def get_input_luminance_range(self):
        return (0, 1)

    def get_input_range(self):
        return (0, 1)

    def get_output_range(self):
        return self.base.get_output_range()
