import skimage.io as image_io
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern


class LBP:
    def __init__(self, radius=1, n_points=None):
        super().__init__()
        self.radius = radius
        self.n_points = n_points if n_points else 8*radius
        self.method = 'uniform'

    def get_image(self, image):
        lbp_image = local_binary_pattern(image,  self.n_points, self.radius)
        return lbp_image

    def get_histogram(self, lbp):
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(2**self.n_points + 1),
                               density=True)
        hist = hist/sum(hist)
        return hist
