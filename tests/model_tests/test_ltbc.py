import unittest

import numpy

from comicolorization.models import ltbc


class TestLtbc(unittest.TestCase):
    def setUp(self):
        self.model = ltbc.Ltbc()
        self.size_image = 224

    def test_shape(self):
        input = numpy.zeros((1, 1, self.size_image, self.size_image), dtype=numpy.float32)
        output = self.model(input, test=True)
        self.assertEqual(input.shape[0], output.shape[0])
        self.assertEqual(input.shape[2], output.shape[2])
        self.assertEqual(input.shape[3], output.shape[3])
