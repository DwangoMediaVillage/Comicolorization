import chainer
import numpy


class BaseModel(chainer.Chain):
    def __call__(self, x, test: bool = False):
        """
        this method should return images array related loss mode (Lab, RGB, etc)
        """
        raise NotImplementedError

    def generate_rgb_image(self, gray_images: numpy.ndarray, **kwargs):
        raise NotImplementedError
