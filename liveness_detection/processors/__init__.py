import os
import sys


def get_path_for_test(directory=".."):
    dirname = os.path.dirname(__file__)
    relative_path = os.path.join(dirname, directory)
    absolute_path = os.path.abspath(relative_path)
    return absolute_path


sys.path.insert(0, get_path_for_test())
