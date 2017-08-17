import numpy
from PIL import Image
from skimage.color import rgb2lab

import comicolorization
import comicolorization_sr


def _get_input_panel_rect(panel_size, input_width):
    """
    ニューラルネットモデルに入力する画像内の、コマ画像の位置を求める。
    モデルの入力は正方形のみを想定してるのに対し、コマ画像は長方形を想定しているため、この値が必要になる。
    :param panel_size: コマ画像のサイズ。[width, height]
    :param input_width: ニューラルネットモデルが想定している正方形画像の横幅
    :return: コマ画像の位置。[左上x, 左上y, 右下x, 右下y]
    """
    w, h = panel_size
    scale = min(input_width / w, input_width / h)

    w, h = (round(w * scale), round(h * scale))
    x, y = (input_width - w) // 2, (input_width - h) // 2
    return [x, y, x + w, y + h]


def _make_input_panel_image(panel_image, input_panel_rect, input_width):
    """
    ニューラルネットモデルに入力する画像を作成する。
    :param panel_image: コマ画像
    :param input_panel_rect: _get_input_panel_rectで求めた値
    :param input_width: ニューラルネットモデルが想定している正方形画像の横幅
    :return: ニューラルネットモデル用の入力画像
    """
    x, y, _w, _h = input_panel_rect
    w, h = _w - x, _h - y

    img = panel_image.convert('L')
    img = img.resize((w, h), Image.BICUBIC)

    bg = Image.new('RGB', (input_width, input_width), '#ffffff')
    bg.paste(img, (x, y))
    return bg


def _make_binarized_image(img, threshold):
    """
    二値化された画像を作成する
    :param img: 二値化対象の画像
    :param threshold: 閾値
    :return: 二値化された画像
    """
    img = ((numpy.array(img) > threshold) * 255).astype(numpy.uint8)
    img = Image.fromarray(img)
    return img


class PanelPipeline(object):
    """
    １コマ着色パイプライン
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
        :param image: 任意のサイズのコマ画像
        :param reference_image: 任意のサイズの参照画像
        :param resize_width: リサイズ後の正方形の横サイズ。
        :param threshold: 二値化の際の閾値
        """
        self.drawer = drawer
        self.drawer_sr = drawer_sr
        self.image = image
        self.reference_image = reference_image
        self.resize_width = resize_width
        self.threshold = threshold

        self._size_pre = None  # 元の画像サイズ
        self._crop_pre = None  # NN入力画像内のコマ画像の位置

    def process(self):
        """
        着色処理
        """
        small_input_image, big_input_image = self._pre_process()
        drawn_panel_image = self._draw_process(small_input_image, big_input_image)
        return self._post_process(drawn_panel_image)

    def _pre_process(self):
        """
        前処理。
        ・入力画像をリサイズ
        ・二値化
        ・白埋めして正方形に
        """
        self._size_pre = self.image.size

        small_crop_pre = _get_input_panel_rect(
            panel_size=self.image.size,
            input_width=self.resize_width,
        )

        input_panel_image = _make_input_panel_image(
            panel_image=self.image,
            input_panel_rect=small_crop_pre,
            input_width=self.resize_width,
        )

        # 超解像用の処理
        self._crop_pre = _get_input_panel_rect(
            panel_size=self.image.size,
            input_width=self.resize_width * 2,
        )

        # 入力画像の作成
        small_input_image = _make_binarized_image(input_panel_image, self.threshold)
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
        後処理。
        ・白や黒に近い画素値を白や黒にする
        ・埋めた箇所をクロップ
        ・元のサイズにリサイズ
        """
        array = numpy.array(drawn_panel_image)
        th = 255 / 6 / 2
        array[(array < th).all(axis=2)] = numpy.ones(3) * 0
        array[(array > 255 - th).all(axis=2)] = numpy.ones(3) * 255
        image = Image.fromarray(array)

        if self._crop_pre is not None:
            image = image.crop(self._crop_pre)
        if self._size_pre is not None:
            image = image.resize((self._size_pre), Image.BICUBIC)
        return image
