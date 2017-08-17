from abc import ABCMeta, abstractmethod
import numpy
from PIL import Image
from skimage.color import rgb2lab
import typing


class BaseDataProcess(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, data, test):
        pass


class RandomScaleImageProcess(BaseDataProcess):
    def __init__(self, min_scale: float, max_scale: float):
        self._min_scale = min_scale
        self._max_scale = max_scale

    def __call__(self, image: Image.Image, test):
        base_size = image.size

        rand = numpy.random.rand(1) if not test else 0.5

        scale = rand * (self._max_scale - self._min_scale) + self._min_scale
        size_resize = (int(image.size[0] * scale), int(image.size[1] * scale))

        if base_size != size_resize:
            image = image.resize(size_resize, resample=Image.BICUBIC)

        return image


class LabImageArrayProcess(BaseDataProcess):
    def __init__(self, normalize=True, dtype=numpy.float32):
        self._normalize = normalize
        self._dtype = dtype

    def __call__(self, image: Image.Image, test):
        image = numpy.asarray(image, dtype=self._dtype)[:, :, :3] / 255  # rgb
        image = rgb2lab(image).astype(self._dtype).transpose(2, 0, 1)

        if self._normalize:
            image /= 50
            image[0] -= 1

        return image


class ChainProcess(BaseDataProcess):
    def __init__(self, process: typing.Iterable[BaseDataProcess]):
        self._process = process

    def __call__(self, data, test):
        for p in self._process:
            data = p(data, test)
        return data
