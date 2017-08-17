import math
import chainer


class Discriminator(chainer.Chain):
    def __init__(self, size, first_pooling_size=1):
        last_size = size // (2 ** 4) // first_pooling_size

        super().__init__(
            c0=chainer.links.Convolution2D(3, 64, 4, stride=2, pad=1),
            c1=chainer.links.Convolution2D(64, 128, 4, stride=2, pad=1),
            c2=chainer.links.Convolution2D(128, 256, 4, stride=2, pad=1),
            c3=chainer.links.Convolution2D(256, 512, 4, stride=2, pad=1),
            bn0=chainer.links.BatchNormalization(64),
            bn1=chainer.links.BatchNormalization(128),
            bn2=chainer.links.BatchNormalization(256),
            bn3=chainer.links.BatchNormalization(512),
            l0z=chainer.functions.Linear(last_size ** 2 * 512, 1, wscale=0.02 * math.sqrt(last_size ** 2 * 512)),
        )

        if first_pooling_size > 1:
            self.first_pooling = chainer.functions.AveragePooling2D(first_pooling_size, stride=first_pooling_size)
        else:
            self.first_pooling = lambda x: x  # through pass

    def __call__(self, x, test=False):
        h = self.first_pooling(x)
        h = chainer.functions.relu(self.c0(h))
        h = chainer.functions.relu(self.bn1(self.c1(h), test=test))
        h = chainer.functions.relu(self.bn2(self.c2(h), test=test))
        h = chainer.functions.relu(self.bn3(self.c3(h), test=test))
        l = self.l0z(h)
        return l
