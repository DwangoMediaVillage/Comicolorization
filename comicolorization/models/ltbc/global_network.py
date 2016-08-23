import chainer


class GlobalNetwork(chainer.Chain):
    def __init__(self, use_classification=False):
        super().__init__(
            conv1_1=chainer.functions.Convolution2D(
                in_channels=512,
                out_channels=512,
                ksize=3,
                stride=2,
                pad=1),
            bn1_1=chainer.links.BatchNormalization(512),
            conv1_2=chainer.functions.Convolution2D(
                in_channels=None,
                out_channels=512,
                ksize=3,
                stride=1,
                pad=1),
            bn1_2=chainer.links.BatchNormalization(512),

            conv2_1=chainer.functions.Convolution2D(
                in_channels=None,
                out_channels=512,
                ksize=3,
                stride=2,
                pad=1),
            bn2_1=chainer.links.BatchNormalization(512),
            conv2_2=chainer.functions.Convolution2D(
                in_channels=None,
                out_channels=512,
                ksize=3,
                stride=1,
                pad=1),
            bn2_2=chainer.links.BatchNormalization(512),

            l3_1=chainer.links.Linear(7 * 7 * 512, 1024),
            bn3_1=chainer.links.BatchNormalization(1024),
            l3_2=chainer.links.Linear(None, 512),
            bn3_2=chainer.links.BatchNormalization(512),
            l3_3=chainer.links.Linear(None, 256),
            bn3_3=chainer.links.BatchNormalization(256),
        )

        self.use_classification = use_classification

    def __call__(self, x, test: bool = False):
        h = x
        h = chainer.functions.relu(self.bn1_1(self.conv1_1(h), test=test))
        h = chainer.functions.relu(self.bn1_2(self.conv1_2(h), test=test))
        h = chainer.functions.relu(self.bn2_1(self.conv2_1(h), test=test))
        h = chainer.functions.relu(self.bn2_2(self.conv2_2(h), test=test))
        h = chainer.functions.relu(self.bn3_1(self.l3_1(h), test=test))
        h = h_for_classification = chainer.functions.relu(self.bn3_2(self.l3_2(h), test=test))
        h = chainer.functions.relu(self.bn3_3(self.l3_3(h), test=test))

        if not self.use_classification:
            return h
        else:
            return h, h_for_classification
