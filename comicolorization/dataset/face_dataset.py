import cv2
import numpy
from PIL import Image

from .image_dataset import PILImageDatasetBase


class FaceImageDataset(PILImageDatasetBase):
    """
    Dataset of cropped facial region

    This dataset reads an external image file on every call.
    Then extract facial region with cascade classifier.

    This dataset provides None as Image if no faces are found.
    You need to ignore None with your script.
    """

    def __init__(self, paths, classifier_path, input_resize=None, output_resize=None, root='.', margin_ratio=0.3):
        """
        :param paths: image files :see: https://github.com/pfnet/chainer/blob/master/chainer/datasets/image_dataset.py
        :param classifier_path: XML of pre-trained face detector.
        You can find it from https://github.com/opencv/opencv/tree/master/data/haarcascades
        :param input_resize: set it if you want to resize image **before** running face detector
        :param output_resize: target size of output image
        """
        super().__init__(paths=paths, resize=input_resize, root=root)
        self.classifier = cv2.CascadeClassifier(classifier_path)
        self.margin_ratio = margin_ratio
        self.output_resize = output_resize

    def get_example(self, i) -> (str, Image):
        path, image = super().get_example(i)
        image_array = numpy.asarray(image)
        image_height, image_width = image_array.shape[:2]
        if len(image_array.shape) == 2:  # gray image
            gray = image_array
        else:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        facerects = self.classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(64, 64))
        if len(facerects) == 0:
            return path, None  # more sophisticated way to handle errors?
        x, y, width, _ = facerects[0]
        margin = int(width * self.margin_ratio)
        if min(
                y, image_height - y - width,
                x, image_width - x - width,
        ) < margin:  # cannot crop
            return path, None

        cropped = image_array[y - margin:y + width + margin, x - margin:x + width + margin]
        if self.output_resize is None:
            return path, Image.fromarray(cropped)
        else:
            return path, Image.fromarray(cropped).resize(self.output_resize)
