import chainer
import typing

from comicolorization_sr.config import ModelConfig
from comicolorization_sr.model import Unet


class Forwarder(chainer.ChainList):
    def __init__(
            self,
            config: ModelConfig,
            colorizer: typing.Callable[[typing.Any, bool], typing.Any],
            model: Unet,
    ):
        super().__init__(config)
        self.config = config
        self.colorizer = colorizer
        self.model = model

    def __call__(self, input, concat, test):
        return self.forward(input, concat, test)

    def forward(self, input, concat, test):
        image = self.forward_colorizer(input, test)['image']
        outputs = self.forward_super_pixel(image, concat, test)
        return outputs

    def forward_colorizer(self, input, test):
        # generate smaller image
        image = self.colorizer(input, test=test)
        if isinstance(image, chainer.Variable):
            image.unchain_backward()

        return {
            'image': image
        }

    def forward_super_pixel(self, image, concat, test):
        # make super pixel input image
        image = chainer.functions.unpooling_2d(image, ksize=self.config.scale, cover_all=False)
        image = chainer.functions.concat((concat, image), axis=1)

        # generate super pixel image
        image, _ = self.model(image, test=test)
        outputs = {
            'image': image,
        }

        return outputs
