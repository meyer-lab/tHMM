""" Unit test file. """
import unittest

from ..LineageInputOutput import import_Heiser, tryRecursion


class TestModel(unittest.TestCase):
    """
    Unit test class for importing data.
    """
    def setUp(self):
        """
        Gets the path to synthetic data.
        This data is formatted similarly to Heiser lab data,
        but contains known mistakes/exceptions that the functions
        should be able to handle.
        """
        self.path_to_synthetic_data = #TODO
        
    def test_import_Heiser(self):
        """
        Tests the main import function for Heiser lab data.
        """
        path2use = self.path_to_synthetic_data
        
        
    def test_tryRecursion(self):
        """
        Tests the recursion function used to recurse acros Excel cells
        in Heiser lab data.
        """
        path2use = self.path_to_synthetic_data
