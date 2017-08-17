import numpy
from PIL import Image
import PIL.ImageOps
import PIL.ImageFilter
import PIL.ImageChops
import typing

import comicolorization
import comicolorization_sr
from . import PanelPipeline


def _make_rawline_image(bw_image, filter_size=5):
    """
    Dilate+Diffを使って、擬似的に線画を作る。
    :param bw_image: 元画像
    :param filter_size: 最大値フィルタのサイズ
    :return: 線画
    """
    bw = bw_image.convert('L')
    line_raw = bw.filter(PIL.ImageFilter.MaxFilter(filter_size))  # opencvのdilateの代わり
    line_raw = PIL.ImageChops.difference(bw, line_raw)
    line_raw = PIL.ImageOps.invert(line_raw)
    return line_raw


def _make_panel_image(base_image, panel_rect):
    """
    コマ位置を指定してコマ画像を切り出す。
    :param base_image: 元画像
    :param panel_rect: コマ位置。[左上x,左上y,width,height]
    :return: 切り出されたコマ画像
    """
    width = panel_rect[2]
    height = panel_rect[3]
    img = base_image.crop((panel_rect[0], panel_rect[1], panel_rect[0] + width, panel_rect[1] + height))
    return img


def _make_page_image(base_image, panel_image_list, offset_list):
    """
    ページ画像から切り出したコマ画像を、元の位置に戻して、ページ画像を合成する。
    :param base_image: 元のページ画像
    :param panel_image_list: コマ画像のリスト
    :param offset_list: コマ画像の左上位置のリスト。[[x,y], ...]
    :return: 合成したページ画像
    """
    base = base_image.copy()
    for panel_image, offset in zip(panel_image_list, offset_list):
        base.paste(panel_image, tuple(offset))
    return base


def _make_line_image(rawline_image, threshould=150):
    """
    二値化された線画を作成する。
    :param rawline_image: _make_rawline_imageで作成した線画
    :param threshould: 二値化の閾値
    :return: 二値化された線画
    """
    line = (numpy.array(rawline_image) > threshould).astype(numpy.uint8) * 255
    line = Image.fromarray(line)
    return line


def _make_overlay_image(page_image, line_image):
    """
    post process。
    :param page_image: ページ画像
    :param line_image: ページ画像に重ねる線画
    :return: post process後のページ画像
    """
    img = page_image.copy()
    alpha = PIL.ImageOps.invert(line_image)
    img.paste(line_image, mask=alpha)
    return img


class PagePipeline(object):
    """
    １ページの、着色以外のパイプライン
    """

    def __init__(
            self,
            drawer: comicolorization.drawer.Drawer,
            drawer_sr: comicolorization_sr.drawer.Drawer,
            image,
            reference_images,
            threshold_binary,
            threshold_line,
            panel_rects,
    ):
        """
        :param image: 任意のサイズのページ画像
        :param reference_images: 任意のサイズの参照画像。コマの数だけ必要
        :param threshold_binary: コマ二値化の際の閾値
        :param threshold_line: 重ねる線画の閾値
        :param panel_rects: コマ位置の配列。[[左上x,左上y,width,height], ...]
        """
        self.drawer = drawer
        self.drawer_sr = drawer_sr
        self.image = image
        self.reference_images = reference_images
        self.panel_rects = panel_rects
        self.threshold_binary = threshold_binary
        self.threshold_line = threshold_line

        self._raw_image = image
        self._raw_line = _make_rawline_image(image)  # 線画

    def process(self):
        """
        着色処理
        """
        panels = self._pre_process()
        drawn_panel_images = [panel.process() for panel in panels]
        return self._post_process(drawn_panel_images)

    def _pre_process(self) -> typing.List[PanelPipeline]:
        """
        前処理。
        ・コマ分割
        """
        panels = []
        for reference_image, panel_rect in zip(self.reference_images, self.panel_rects):
            bw_panel = _make_panel_image(self._raw_image, panel_rect)
            panel = PanelPipeline(
                drawer=self.drawer,
                drawer_sr=self.drawer_sr,
                image=bw_panel,
                reference_image=reference_image,
                threshold=self.threshold_binary,
            )
            panels.append(panel)

        return panels

    def _post_process(self, drawn_panel_images):
        """
        後処理。
        ・コマごとに後処理
        ・１ページに戻す
        ・線画を重ねる
        """
        # 漫画のコマを着色済みのものに置き換え
        bg = _make_page_image(self._raw_image, drawn_panel_images, [[r[0], r[1]] for r in self.panel_rects])

        # オーバーレイ用の線画を作成
        line = _make_line_image(self._raw_line, self.threshold_line)

        # 合成
        output = _make_overlay_image(bg, line)
        return output
