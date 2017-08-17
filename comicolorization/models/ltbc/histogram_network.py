from typing import Tuple

import chainer
import numpy


class HistogramNetwork(chainer.Chain):
    def __init__(
            self,
            ranges: Tuple[Tuple[int, int], ...],
            num_bins: int,
            threshold_palette: float,
            use_multidimensional=False,
            dtype=numpy.float32,
    ):
        """
        if not multidimensional: output size will be [batch, num_bins*channel]
        if multidimensional: output size will be [batch, num_bins^channel]
        :param ranges: color range for each channel
        :param num_bins: number of bins for each channel
        :param threshold_palette: the threshold of palette mode. if None, then histogram mode.
        :param use_multidimensional: if it is True, multidimensional histogram mode
        """
        super().__init__()

        self.ranges = ranges
        self.num_bins = num_bins
        self.threshold_palette = threshold_palette
        self.use_multidimensional = use_multidimensional
        self.dtype = dtype

    def __call__(self, h, test=False):
        """
        :param h: (batchsize, channel, size x, size y)
        :return: (batchsize, channel*num_bins)
        """
        if isinstance(h, chainer.Variable):
            h = h.data

        batchsize, channel, x, y = h.shape

        xp = self.xp
        if self._cpu:
            h = chainer.cuda.to_cpu(h)
        else:
            h = chainer.cuda.to_gpu(h)

        histogram_list = []
        for h_one in h:
            if not self.use_multidimensional:
                histogram_one = []
                for h_channel, _range in zip(h_one, self.ranges):
                    array = (h_channel - _range[0]) / (_range[1] - _range[0]) * self.num_bins
                    array = xp.reshape(array, -1).astype(numpy.int32)
                    array = xp.where(array == self.num_bins, array - 1, array)
                    hist = xp.bincount(array, minlength=self.num_bins)
                    histogram_one.append(hist)

                histogram_one = xp.reshape(xp.concatenate(histogram_one, axis=0), (1, -1))

            else:
                array = xp.empty(h_one.shape, numpy.int32)
                for i, _range in zip(range(channel), self.ranges):
                    array[i] = (h_one[i] - _range[0]) / (_range[1] - _range[0]) * self.num_bins
                    array[i] = xp.where(array[i] == self.num_bins, array[i] - 1, array[i])

                array = array[0] * self.num_bins * self.num_bins + array[1] * self.num_bins + array[2]
                array = xp.reshape(array, -1)
                histogram_one = xp.bincount(array, minlength=self.num_bins ** channel)

            histogram_list.append(histogram_one)

        h = xp.concatenate(histogram_list, axis=0) / (x * y)
        if self.threshold_palette is not None:
            h_palette = xp.where(h > self.threshold_palette, xp.ones_like(h), xp.zeros_like(h))
            h = h_palette

        h = h.astype(self.dtype)
        h = h.reshape((batchsize, -1))

        h = chainer.Variable(h, volatile=test)
        return h
