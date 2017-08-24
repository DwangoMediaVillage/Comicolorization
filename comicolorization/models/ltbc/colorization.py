import chainer


class ColorizationNetwork(chainer.Chain):
    def __init__(self, output_channels=2):
        super(ColorizationNetwork, self).__init__(
            conv1_1=chainer.links.Convolution2D(
                in_channels=256,
                out_channels=128,
                ksize=3,
                stride=1,
                pad=1,
            ),
            bn1_1=chainer.links.BatchNormalization(128),

            conv2_1=chainer.links.Convolution2D(
                in_channels=None,
                out_channels=64,
                ksize=3,
                stride=1,
                pad=1,
            ),
            bn2_1=chainer.links.BatchNormalization(64),
            conv2_2=chainer.links.Convolution2D(
                in_channels=None,
                out_channels=64,
                ksize=3,
                stride=1,
                pad=1,
            ),
            bn2_2=chainer.links.BatchNormalization(64),

            conv3_1=chainer.links.Convolution2D(
                in_channels=None,
                out_channels=32,
                ksize=3,
                stride=1,
                pad=1,
            ),
            bn3_1=chainer.links.BatchNormalization(32),
            conv3_2=chainer.links.Convolution2D(
                in_channels=None,
                out_channels=output_channels,
                ksize=3,
                stride=1,
                pad=1,
            ),
        )

    def __call__(self, x, test=False):
        # type: (any, bool) -> any
        h = x
        h = chainer.functions.relu(self.bn1_1(self.conv1_1(h), test=test))
        h = chainer.functions.unpooling_2d(h, ksize=2, cover_all=False)
        h = chainer.functions.relu(self.bn2_1(self.conv2_1(h), test=test))
        h = chainer.functions.relu(self.bn2_2(self.conv2_2(h), test=test))
        h = chainer.functions.unpooling_2d(h, ksize=2, cover_all=False)
        h = chainer.functions.relu(self.bn3_1(self.conv3_1(h), test=test))
        h = chainer.functions.sigmoid(self.conv3_2(h))
        return h
