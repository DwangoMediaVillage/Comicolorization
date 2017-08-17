import numpy
from PIL import Image
from skimage.color import rgb2lab

import comicolorization
import comicolorization_sr
from .process import make_binarized_image


def _calc_input_panel_rect(panel_size, input_width):
    """
    Calc rectangle of the panel image for inputting to neural network model.
    Because panel image isn't square but neural network model postulate square
    :param panel_size: size of source panel image [width, height]
    :param input_width: width of input image for neural network model
    :return: rectangle of panel image [left, top, right, bottom]
    """
    w, h = panel_size
    scale = min(input_width / w, input_width / h)

    w, h = (round(w * scale), round(h * scale))
    x, y = (input_width - w) // 2, (input_width - h) // 2
    return [x, y, x + w, y + h]


def _make_input_panel_image(panel_image, input_panel_rect, input_width):
    """
    Make input image for neural network model
    :param panel_image: source panel image
    :param input_panel_rect: rectangle calculated by _calc_input_panel_rect
    :param input_width: width of input image for neural network model
    :return: input image for neural network model
    """
    x, y, _w, _h = input_panel_rect
    w, h = _w - x, _h - y

    img = panel_image.convert('L')
    img = img.resize((w, h), Image.BICUBIC)

    bg = Image.new('RGB', (input_width, input_width), '#ffffff')
    bg.paste(img, (x, y))
    return bg


class PanelPipeline(object):
    """
    The pipeline of one panel
    """

    def __init__(
            self,
            drawer: comicolorization.drawer.Drawer,
            drawer_sr: comicolorization_sr.drawer.Drawer,
            image,
            reference_image,
            resize_width=224,
            threshold=200,
    ):
        """
        :param drawer: drawer of the comicolorization task
        :param drawer_sr: draw of the super resolution task
        :param image: source panel iamge
        :param reference_image: reference image
        :param resize_width: width of input image for neural network model
        :param threshold: threshold by using binarizing input image
        """
        self.drawer = drawer
        self.drawer_sr = drawer_sr
        self.image = image
        self.reference_image = reference_image
        self.resize_width = resize_width
        self.threshold = threshold

        self._crop_pre = None  # rectangle of panel image in input image

    def process(self):
        """
        colorization process
        """
        small_input_image, big_input_image = self._pre_process()
        drawn_panel_image = self._draw_process(small_input_image, big_input_image)
        return self._post_process(drawn_panel_image)

    def _pre_process(self):
        """
        * resize panel image
        * binarization
        * padding and make square image
        """
        small_crop_pre = _calc_input_panel_rect(
            panel_size=self.image.size,
            input_width=self.resize_width,
        )

        input_panel_image = _make_input_panel_image(
            panel_image=self.image,
            input_panel_rect=small_crop_pre,
            input_width=self.resize_width,
        )

        self._crop_pre = _calc_input_panel_rect(
            panel_size=self.image.size,
            input_width=self.resize_width * 2,
        )

        small_input_image = make_binarized_image(input_panel_image, self.threshold)
        big_input_image = _make_input_panel_image(self.image, self._crop_pre, self.resize_width * 2)

        return small_input_image, big_input_image

    def _draw_process(self, small_input_image, big_input_image):
        concat_image_process = self.drawer_sr.colorization.get_concat_process()

        lab = rgb2lab(numpy.array(small_input_image))
        lab[:, :, 0] /= 100
        small_image = self.drawer.draw(
            input_images_array=lab.astype(numpy.float32).transpose(2, 0, 1)[numpy.newaxis],
            rgb_images_array=numpy.array(self.reference_image, dtype=numpy.float32).transpose(2, 0, 1)[numpy.newaxis],
        )[0]

        small_image = small_image.convert('RGB')
        small_array = numpy.array(small_image, dtype=numpy.float64)
        small_array = rgb2lab(small_array / 255).astype(numpy.float32)
        small_array = small_array.transpose(2, 0, 1) / 100

        large_array = concat_image_process(big_input_image, test=True)
        drawn_panel_image = self.drawer_sr.draw_only_super_pixel(
            image=small_array[numpy.newaxis],
            concat=large_array[numpy.newaxis],
        )[0]

        return drawn_panel_image

    def _post_process(self, drawn_panel_image):
        """
        * bring near white/black to white/black
        * crop
        * resize
        """
        array = numpy.array(drawn_panel_image)
        th = 255 / 6 / 2
        array[(array < th).all(axis=2)] = numpy.ones(3) * 0
        array[(array > 255 - th).all(axis=2)] = numpy.ones(3) * 255
        image = Image.fromarray(array)

        image = image.crop(self._crop_pre)
        image = image.resize(self.image.size, Image.BICUBIC)
        return image
