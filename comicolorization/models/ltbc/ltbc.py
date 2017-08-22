import chainer
import numpy

from .low_level import LowLevelNetwork
from .mid_level import MidLevelNetwork
from .colorization import ColorizationNetwork
from .global_network import GlobalNetwork
from .classification_network import ClassificationNetwork
from .fusion_layer import FusionLayer
from ..base_model import BaseModel
from ...utility.image import array_to_image
from .histogram_network import HistogramNetwork
from ..normalize import ColorNormalize


class Ltbc(BaseModel):
    def __init__(
            self,
            use_global=True,
            use_classification=False,
            classification_num_output_list=None,
            use_histogram=False,
            use_multidimensional_histogram=False,
            num_bins_histogram=None,
            threshold_histogram_palette=None,
            reinput_mode=None,
            loss_type='RGB',
    ):
        """
        :param use_global: if True, use global feature network
        :param use_classification: if True, use classification network
        :param use_histogram: if True, use histogram network
        :param use_multidimensional_histogram: if it is True, multidimensional histogram mode
        :param threshold_histogram_palette: the threshold of palette mode. if None, then histogram mode.
        :param reinput_mode:
            None: no reinput
            color: can input 3 channel image
        """
        if use_multidimensional_histogram:
            assert use_histogram, "when using multidimensional histogram, should set `use_histogram=True`"

        out_channels = 2 if loss_type == 'ab' else 3
        super(Ltbc,self).__init__(
            low_level=LowLevelNetwork(),
            mid_level=MidLevelNetwork(),
            fusion_layer=FusionLayer(),
            colorization=ColorizationNetwork(output_channels=out_channels),
        )

        self.use_global = use_global
        self.use_classification = use_classification
        self.use_histogram = use_histogram
        self.reinput_mode = reinput_mode
        self.loss_type = loss_type
        self.out_channels = out_channels

        if self.use_global:
            self.add_link("global_network", GlobalNetwork(use_classification=use_classification))

        if self.use_classification:
            self.add_link(
                "classification_network",
                ClassificationNetwork(num_output_list=classification_num_output_list),
            )

        if self.use_histogram:
            ranges = ((0, 255), (0, 255), (0, 255))
            self.add_link("histogram_network", HistogramNetwork(
                ranges=ranges,
                num_bins=num_bins_histogram,
                threshold_palette=threshold_histogram_palette,
                use_multidimensional=use_multidimensional_histogram,
            ))

    def __call__(self, x, x_global=None, x_rgb=None, x_histogram=None, test=False):
        # type: (any, any, any, any, bool) -> any
        """
        :param x: input image
        :param x_global: global input image. if None, then x_global=x
        :param x_rgb: reference image for color feature
        :param x_histogram: color histogram. if it is None, then calc color histogram by reference image
        """
        if self.use_histogram:
            assert x_rgb is not None or x_histogram is not None, "must give `x_rgb` or `x_histogram`"
            assert x_rgb is None or x_histogram is None, "cannot give `x_rgb` and `x_histogram`"

        if self.reinput_mode is None:
            pass
        elif self.reinput_mode == 'color':
            # if gray image, zero padding
            if x.shape[1] == 1:
                zeropad = self.xp.zeros((x.shape[0], 2, x.shape[2], x.shape[3]), dtype=x.dtype)
                x = chainer.functions.concat((x, zeropad), axis=1)
        else:
            raise NotImplementedError

        if x_global is None:
            x_global = x

        h = x
        h = self.low_level(h, test=test)
        h = self.mid_level(h, test=test)

        h_classification = None

        one_dimension_feature_list = []
        if self.use_global:
            h_global = x_global
            h_global = self.low_level(h_global, test=test)

            if not self.use_classification:
                h_global = self.global_network(h_global, test=test)
            else:
                h_global, h_for_classification = self.global_network(h_global, test=test)
                h_classification = self.classification_network(h_for_classification, test=test)

            one_dimension_feature_list.append(h_global)

        if self.use_histogram:
            if x_histogram is None:
                h_histogram = self.histogram_network(x_rgb, test=test)
            else:
                h_histogram = chainer.Variable(self.xp.array(x_histogram.reshape((x.shape[0], -1))))
            one_dimension_feature_list.append(h_histogram)

        if self.use_global or self.use_histogram:
            h = self.fusion_layer(h=h, one_dimension_feature_list=one_dimension_feature_list, test=test)

        h = self.colorization(h, test=test)
        h = h_before_sigmoid = chainer.functions.unpooling_2d(h, ksize=2, cover_all=False)

        in_min = (0, 0, 0)
        in_max = (1, 1, 1)
        h = ColorNormalize(self.loss_type, in_min=in_min, in_max=in_max)(h)

        return h, {
            'classification': h_classification,
            'before_sigmoid': h_before_sigmoid,
        }

    def generate_rgb_image(self, gray_images_array, rgb_images_array=None, histogram_array=None):
        color_images_array = self(gray_images_array, x_rgb=rgb_images_array, x_histogram=histogram_array, test=True)

        color_images_array = color_images_array[0]

        images = array_to_image(
            color_images_array.data,
            gray_images_array=gray_images_array,
            mode=self.loss_type,
        )
        return images
