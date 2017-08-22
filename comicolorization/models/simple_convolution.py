import chainer
import numpy

from .base_model import BaseModel
from ..utility.image import array_to_image


class SimpleConvolution(BaseModel):
    def __init__(self, loss_type='RGB'):
        out_channels = 2 if loss_type == 'ab' else 3
        super(SimpleConvolution,self).__init__(
            conv1=chainer.functions.Convolution2D(
                in_channels=1,
                out_channels=64,
                ksize=7,
                stride=1,
                pad=3),
            conv2=chainer.functions.Convolution2D(
                in_channels=None,
                out_channels=128,
                ksize=7,
                stride=1,
                pad=3),
            fc1=chainer.functions.Convolution2D(
                in_channels=None,
                out_channels=64,
                ksize=1,
                stride=1,
                pad=0),
            fc2=chainer.functions.Convolution2D(
                in_channels=None,
                out_channels=out_channels,
                ksize=1,
                stride=1,
                pad=0),
        )

        self.loss_type = loss_type
        self.out_channels = out_channels

    def generate_rgb_image(self, gray_images_array, **kwargs):
        # type: (any, numpy.ndarray, **any) -> any
        color_images_array = self(gray_images_array, test=True).data
        images = array_to_image(
            color_images_array,
            gray_images_array=gray_images_array,
            mode=self.loss_type,
        )
        return images

    def __call__(self, x, test=False):
        # type: (chainer.Variable, bool) -> any
        h = x
        h = chainer.functions.relu(self.conv1(h))
        h = chainer.functions.relu(self.conv2(h))
        h = chainer.functions.relu(self.fc1(h))
        h = self.fc2(h)
        return h
