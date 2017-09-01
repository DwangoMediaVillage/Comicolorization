from abc import ABCMeta, abstractmethod
import chainer
import typing
import six

from comicolorization_sr.config import ModelConfig
from comicolorization_sr import utility

@six.add_metaclass(ABCMeta)
class BaseModel(chainer.Chain, object):
    def __init__(self, config, **kwargs):
        # type: (ModelConfig, **any) -> None
        super(BaseModel, self).__init__(**kwargs)
        self.config = config

    @abstractmethod
    def __call__(self, x, test):
        # type: (any, any) -> typing.Tuple[chainer.Variable, typing.Dict]
        pass


class Unet(BaseModel):
    def __init__(self, config):
        # type: (ModelConfig) -> None
        super(Unet, self).__init__(
            config,
            c0=utility.chainer_utility.Link.create_convolution_2d(4, 32, 3, 1, 1),
            c1=utility.chainer_utility.Link.create_convolution_2d(32, 64, 4, 2, 1),
            c2=utility.chainer_utility.Link.create_convolution_2d(64, 64, 3, 1, 1),
            c3=utility.chainer_utility.Link.create_convolution_2d(64, 128, 4, 2, 1),
            c4=utility.chainer_utility.Link.create_convolution_2d(128, 128, 3, 1, 1),
            c5=utility.chainer_utility.Link.create_convolution_2d(128, 256, 4, 2, 1),
            c6=utility.chainer_utility.Link.create_convolution_2d(256, 256, 3, 1, 1),
            c7=utility.chainer_utility.Link.create_convolution_2d(256, 512, 4, 2, 1),
            c8=utility.chainer_utility.Link.create_convolution_2d(512, 512, 3, 1, 1),

            dc8=utility.chainer_utility.Link.create_deconvolution_2d(1024, 512, 4, 2, 1),
            dc7=utility.chainer_utility.Link.create_convolution_2d(512, 256, 3, 1, 1),
            dc6=utility.chainer_utility.Link.create_deconvolution_2d(512, 256, 4, 2, 1),
            dc5=utility.chainer_utility.Link.create_convolution_2d(256, 128, 3, 1, 1),
            dc4=utility.chainer_utility.Link.create_deconvolution_2d(256, 128, 4, 2, 1),
            dc3=utility.chainer_utility.Link.create_convolution_2d(128, 64, 3, 1, 1),
            dc2=utility.chainer_utility.Link.create_deconvolution_2d(128, 64, 4, 2, 1),
            dc1=utility.chainer_utility.Link.create_convolution_2d(64, 32, 3, 1, 1),
            dc0=utility.chainer_utility.Link.create_convolution_2d(64, 3, 3, 1, 1),

            bnc0=chainer.links.BatchNormalization(32),
            bnc1=chainer.links.BatchNormalization(64),
            bnc2=chainer.links.BatchNormalization(64),
            bnc3=chainer.links.BatchNormalization(128),
            bnc4=chainer.links.BatchNormalization(128),
            bnc5=chainer.links.BatchNormalization(256),
            bnc6=chainer.links.BatchNormalization(256),
            bnc7=chainer.links.BatchNormalization(512),
            bnc8=chainer.links.BatchNormalization(512),

            bnd8=chainer.links.BatchNormalization(512),
            bnd7=chainer.links.BatchNormalization(256),
            bnd6=chainer.links.BatchNormalization(256),
            bnd5=chainer.links.BatchNormalization(128),
            bnd4=chainer.links.BatchNormalization(128),
            bnd3=chainer.links.BatchNormalization(64),
            bnd2=chainer.links.BatchNormalization(64),
            bnd1=chainer.links.BatchNormalization(32)
        )

    def __call__(self, x, test):
        relu = chainer.functions.relu
        concat = chainer.functions.concat

        e0 = relu(self.bnc0(self.c0(x), test=test))
        e1 = relu(self.bnc1(self.c1(e0), test=test))
        e2 = relu(self.bnc2(self.c2(e1), test=test))
        del e1
        e3 = relu(self.bnc3(self.c3(e2), test=test))
        e4 = relu(self.bnc4(self.c4(e3), test=test))
        del e3
        e5 = relu(self.bnc5(self.c5(e4), test=test))
        e6 = relu(self.bnc6(self.c6(e5), test=test))
        del e5
        e7 = relu(self.bnc7(self.c7(e6), test=test))
        e8 = relu(self.bnc8(self.c8(e7), test=test))

        d8 = relu(self.bnd8(self.dc8(concat([e7, e8])), test=test))
        del e7, e8
        d7 = relu(self.bnd7(self.dc7(d8), test=test))
        del d8
        d6 = relu(self.bnd6(self.dc6(concat([e6, d7])), test=test))
        del d7, e6
        d5 = relu(self.bnd5(self.dc5(d6), test=test))
        del d6
        d4 = relu(self.bnd4(self.dc4(concat([e4, d5])), test=test))
        del d5, e4
        d3 = relu(self.bnd3(self.dc3(d4), test=test))
        del d4
        d2 = relu(self.bnd2(self.dc2(concat([e2, d3])), test=test))
        del d3, e2
        d1 = relu(self.bnd1(self.dc1(d2), test=test))
        del d2
        d0 = self.dc0(concat([e0, d1]))

        return d0, {}


def prepare_model(config):
    # type: (ModelConfig) -> any
    model = Unet(config)
    return model
