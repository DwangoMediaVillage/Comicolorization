import chainer
import numpy


class BaseModel(chainer.Chain):
    def __call__(self, x, test=False):
        # type: (any, bool) -> None
        """
        this method should return images array related loss mode (Lab, RGB, etc)
        """
        raise NotImplementedError

    def generate_rgb_image(self, gray_images, **kwargs):
        # type: (numpy.ndarray, **any) -> None
        raise NotImplementedError
