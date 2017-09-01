from abc import ABCMeta, abstractmethod
import numpy
from PIL import Image
from skimage.color import rgb2lab
import typing
import six

@six.add_metaclass(ABCMeta)
class BaseDataProcess(object):
    @abstractmethod
    def __call__(self, data, test):
        pass


class RandomScaleImageProcess(BaseDataProcess):
    def __init__(self, min_scale, max_scale):
        # type: (float, float) -> None
        self._min_scale = min_scale
        self._max_scale = max_scale

    def __call__(self, image, test):
        # type: (Image.Image, any) -> any
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

    def __call__(self, image, test):
        # type: (Image.Image, any) -> any
        image = numpy.asarray(image, dtype=self._dtype)[:, :, :3] / 255  # rgb
        image = rgb2lab(image).astype(self._dtype).transpose(2, 0, 1)

        if self._normalize:
            image /= 50
            image[0] -= 1

        return image


class ChainProcess(BaseDataProcess):
    def __init__(self, process):
        # type: (typing.Iterable[BaseDataProcess]) -> None
        self._process = process

    def __call__(self, data, test):
        for p in self._process:
            data = p(data, test)
        return data
