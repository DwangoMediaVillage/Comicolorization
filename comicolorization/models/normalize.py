import typing
import chainer
from comicolorization import utility


class ColorNormalize(chainer.Chain):
    def __init__(self, type: str, in_min: typing.Tuple, in_max: typing.Tuple):
        super().__init__()
        self.type = type
        self.in_min = in_min
        self.in_max = in_max

    def __call__(self, h):
        in_min = self.in_min
        in_max = self.in_max

        if self.type == 'RGB':
            out_min = (0, 0, 0)
            out_max = (255, 255, 255)
        elif self.type == 'Lab':
            out_min, out_max = utility.color.lab_min_max
        elif self.type == 'ab':
            out_min, out_max = utility.color.lab_min_max
            in_min = self.in_min[1:]
            in_max = self.in_max[1:]
            out_min = out_min[1:]
            out_max = out_max[1:]
        else:
            raise ValueError(self.type)

        h_channels = tuple(
            utility.color.normalize(h_channel, in_min[i], in_max[i], out_min[i], out_max[i])
            for i, h_channel in enumerate(chainer.functions.split_axis(h, h.shape[1], axis=1, force_tuple=True))
        )
        h = chainer.functions.concat(h_channels, axis=1)
        return h
