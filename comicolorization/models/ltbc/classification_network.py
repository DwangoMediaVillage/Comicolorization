import chainer
import typing


class ClassificationNetwork(chainer.Chain):
    """
    linear -> batch normalization -> linear -> batch normalization -> ... -> linear
    """

    def __init__(self, num_output_list):
        # type: (typing.List[int]) -> any
        super(ClassificationNetwork, self).__init__()
        self.num_output_list = num_output_list

        for i, num_output in enumerate(num_output_list):
            self.add_link('l{}'.format(i), chainer.links.Linear(None, num_output))
            if i < len(num_output_list) - 1:
                self.add_link('bn{}'.format(i), chainer.links.BatchNormalization(num_output))

    def __call__(self, h, test=False):
        # type: (any, bool) -> any
        for i, num_output in enumerate(self.num_output_list):
            h = getattr(self, 'l{}'.format(i))(h)

            if i < len(self.num_output_list) - 1:
                h = getattr(self, 'bn{}'.format(i))(h, test=test)

        return h
