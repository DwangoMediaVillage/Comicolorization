import chainer

from comicolorization_sr.config import LossConfig
from comicolorization_sr.forwarder import Forwarder
from comicolorization_sr.model import Unet


class LossMaker(object):
    def __init__(
            self,
            config: LossConfig,
            forwarder: Forwarder,
            model: Unet,
    ):
        self.config = config
        self.forwarder = forwarder
        self.model = model

    @staticmethod
    def blend_loss(loss, blend_config):
        assert sorted(loss.keys()) == sorted(blend_config.keys()), '{} {}'.format(loss.keys(), blend_config.keys())

        sum_loss = None

        for key in sorted(loss.keys()):
            blend = blend_config[key]
            if blend == 0.0:
                continue

            l = loss[key] * blend_config[key]

            if sum_loss is None:
                sum_loss = l
            else:
                sum_loss += l

        return sum_loss

    def make_loss(self, input, concat, target, test):
        output = self.forwarder(input, concat, test)['image']
        mae_loss = chainer.functions.mean_absolute_error(output, target)

        loss = {
            'mae': mae_loss,
        }
        chainer.report(loss, self.model)

        return {
            'main': loss,
        }

    def get_loss_names(self):
        return ['sum_loss'] + list(self.config.blend['main'].keys())

    def sum_loss(self, loss):
        sum_loss = self.blend_loss(loss, self.config.blend['main'])
        chainer.report({'sum_loss': sum_loss}, self.model)
        return sum_loss

    def test(self, input, concat, target):
        loss = self.make_loss(input, concat, target, test=True)
        return self.sum_loss(loss['main'])
