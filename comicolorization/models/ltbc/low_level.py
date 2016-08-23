import chainer


class LowLevelNetwork(chainer.Chain):
    def __init__(self):
        super().__init__(
            conv1_1=chainer.functions.Convolution2D(
                in_channels=None,
                out_channels=64,
                ksize=3,
                stride=2,
                pad=1),
            bn1_1=chainer.links.BatchNormalization(64),
            conv1_2=chainer.functions.Convolution2D(
                in_channels=None,
                out_channels=128,
                ksize=3,
                stride=1,
                pad=1),
            bn1_2=chainer.links.BatchNormalization(128),
            conv2_1=chainer.functions.Convolution2D(
                in_channels=None,
                out_channels=128,
                ksize=3,
                stride=2,
                pad=1),
            bn2_1=chainer.links.BatchNormalization(128),
            conv2_2=chainer.functions.Convolution2D(
                in_channels=None,
                out_channels=256,
                ksize=3,
                stride=1,
                pad=1),
            bn2_2=chainer.links.BatchNormalization(256),
            conv3_1=chainer.functions.Convolution2D(
                in_channels=None,
                out_channels=256,
                ksize=3,
                stride=2,
                pad=1),
            bn3_1=chainer.links.BatchNormalization(256),
            conv3_2=chainer.functions.Convolution2D(
                in_channels=None,
                out_channels=512,
                ksize=3,
                stride=1,
                pad=1),
            bn3_2=chainer.links.BatchNormalization(512),
        )

    def __call__(self, x, test: bool = False):
        h = x
        h = chainer.functions.relu(self.bn1_1(self.conv1_1(h), test=test))
        h = chainer.functions.relu(self.bn1_2(self.conv1_2(h), test=test))
        h = chainer.functions.relu(self.bn2_1(self.conv2_1(h), test=test))
        h = chainer.functions.relu(self.bn2_2(self.conv2_2(h), test=test))
        h = chainer.functions.relu(self.bn3_1(self.conv3_1(h), test=test))
        h = chainer.functions.relu(self.bn3_2(self.conv3_2(h), test=test))
        return h
