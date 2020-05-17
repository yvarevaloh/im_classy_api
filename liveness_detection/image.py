import numpy as np
import skimage.io as image_io
from skimage.color import rgb2gray

from processors.lbp import LBP


class Image:
    _classification_codes = {0: "digital fake", 1: "printed fake", 2: "real"}

    def __init__(self, lbp=None, image_format="png", histogram_format="txt"):
        super().__init__()
        self.lbp = lbp if lbp is not None else LBP()
        self.image_format = image_format
        self.histogram_format = histogram_format

    @staticmethod
    def pre_process_image(image_path):
        return rgb2gray(image_io.imread(image_path))

    def get_lbp_image(self, image_path):
        image = self.pre_process_image(image_path)
        return self.lbp.get_image(image)

    def get_lbp_histogram(self, lbp_image):
        return self.lbp.get_histogram(lbp_image)

    def create_lbp_dataset(self, imagePaths, target_path, machine_learning_path, save_image=True, save_hist=True):
        print("Calculating LBPs")
        for image_path in imagePaths:
            new_path = machine_learning_path + image_path.replace(target_path, "")
            try:
                lbp_image = self.get_lbp_image(image_path)
                if save_image:
                    image_io.imsave(f"{new_path}.{self.image_format}", lbp_image.astype(np.uint8))
                if save_hist:
                    hist = self.get_lbp_histogram(lbp_image)
                    np.savetxt(f"{new_path}.{self.histogram_format}", hist)
            except Exception as e:
                print(f"ERROR: saving {new_path}, \nreason: {e}")
                raise Exception(e)
        print("LBP generation finished")

    def get_classification_code(self, numeric_code=None):
        try:
            if numeric_code is not None and not isinstance(numeric_code, (list, tuple)):
                numeric_code = [numeric_code]
            return [self._classification_codes[code] for code in numeric_code]
        except TypeError as e:
            raise TypeError("numeric_code must be list or tuple.")
        except KeyError as e:
            raise KeyError(f"Classification code {e} doesn't exist.")


# image_instance = Image()
# image_path = r"tests\data\2d.jpg"
# lbp_image = image_instance.get_lbp_image(image_path)
# image_io.imsave(f"{'c:/Users/cristian.narvaez/Documents/git/lDetection/tests/data'}/lbp_image4.png",
#                 lbp_image.astype(np.uint8))
