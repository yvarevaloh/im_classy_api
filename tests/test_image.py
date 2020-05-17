import unittest

from image import Image
from processors.lbp import LBP


class TestImage(unittest.TestCase):

    def setUp(self):
        image_processor = Image()
        image_path = r"tests\data\1r.jpg"
        self.lbp_image = image_processor.get_lbp_image(image_path)
        self.lbp_histogram = image_processor.get_lbp_histogram(self.lbp_image)

    def test_lbp_image(self):
        """Check that the image is in gray"""
        self.assertEqual(len(self.lbp_image.shape), 2)

    def test_lbp_histogram(self):
        """check we have the expected shape in the histogram"""
        radius_expected_size = (256,)
        self.assertEqual(self.lbp_histogram.shape, radius_expected_size)
