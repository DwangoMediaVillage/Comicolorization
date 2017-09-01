import PIL.ImageOps
import PIL.ImageFilter
import PIL.ImageChops
import typing

import comicolorization
import comicolorization_sr
from .panel_pipeline import PanelPipeline
from .process import make_binarized_image


class PagePipeline(object):
    """
    The pipeline of one page
    """

    def __init__(
            self,
            drawer,
            drawer_sr,
            image,
            reference_images,
            threshold_binary,
            threshold_line,
            panel_rects,
    ):
        # type: (comicolorization.drawer.Drawer, comicolorization_sr.drawer.Drawer, any, any, any, any, any) -> None
        """
        :param drawer: drawer of the comicolorization task
        :param drawer_sr: draw of the super resolution task
        :param image: source page image
        :param reference_images: list of reference images
        :param threshold_binary: threshold by using binarizing input image
        :param threshold_line: threshold by using binarizing line-drawing
        :param panel_rects: panel's rectangle [[left,top,width,height], ...]
        """
        self.drawer = drawer
        self.drawer_sr = drawer_sr
        self.image = image
        self.reference_images = reference_images
        self.panel_rects = panel_rects
        self.threshold_binary = threshold_binary
        self.threshold_line = threshold_line

        self._raw_image = image

    def process(self):
        """
        colorization process
        """
        panels = self._pre_process()
        drawn_panel_images = [panel.process() for panel in panels]
        return self._post_process(drawn_panel_images)

    def _pre_process(self):
        # type: (any) -> typing.List[PanelPipeline]
        """
        * split panel images
        """
        panels = []
        for reference_image, panel_rect in zip(self.reference_images, self.panel_rects):
            bw_panel = self._make_panel_image(self._raw_image, panel_rect)
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
        * resynthesis page image
        * overlay line-drawing
        """
        raw_line = self._make_rawline_image(self._raw_image)

        bg = self._make_page_image(self._raw_image, drawn_panel_images, [[r[0], r[1]] for r in self.panel_rects])
        line = make_binarized_image(raw_line, self.threshold_line)

        output = self._make_overlayed_image(bg, line)
        return output

    @staticmethod
    def _make_rawline_image(bw_image, filter_size=5):
        """
        Make line-drawing by using Dilate+Diff
        :param bw_image: source image
        :param filter_size: size of filter
        :return: line-drawing
        """
        bw = bw_image.convert('L')
        line_raw = bw.filter(PIL.ImageFilter.MaxFilter(filter_size))
        line_raw = PIL.ImageChops.difference(bw, line_raw)
        line_raw = PIL.ImageOps.invert(line_raw)
        return line_raw

    @staticmethod
    def _make_panel_image(base_image, panel_rect):
        """
        Make panel image with the panel rectangle
        :param base_image: source image
        :param panel_rect: panel's rectangle [left,top,width,height]
        :return: panel image
        """
        width = panel_rect[2]
        height = panel_rect[3]
        img = base_image.crop((panel_rect[0], panel_rect[1], panel_rect[0] + width, panel_rect[1] + height))
        return img

    @staticmethod
    def _make_page_image(base_image, panel_image_list, offset_list):
        """
        Resynthesis page image with panel images
        :param base_image: source page image
        :param panel_image_list: list of panel images
        :param offset_list: list of panels' left and top [[left,top], ...]
        :return: page image
        """
        base = base_image.copy()
        for panel_image, offset in zip(panel_image_list, offset_list):
            base.paste(panel_image, tuple(offset))
        return base

    @staticmethod
    def _make_overlayed_image(page_image, line_image):
        """
        :param page_image: source page image
        :param line_image: line-drawing for overlay
        :return: image
        """
        img = page_image.copy()
        alpha = PIL.ImageOps.invert(line_image)
        img.paste(line_image, mask=alpha)
        return img
