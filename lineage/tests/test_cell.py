""" Unit test file. """
import unittest
from ../lineage-init/CellNode import CellNode as c, generate

class TestModel(unittest.TestCase):
    """ Here are the unit tests. """

    def test_lifetime(self):
        """ Make sure cell lifetime variables behave properly. """
        cell1 = c.CellNode(key=0, startT=20)
        
        # nan before setting death time
        self.assertTrue(cell1.endT == 'nan')
        self.assertTrue(cell1.tau == 'nan')
        self.assertTrue(cell1.isUnfinished())
        
        # correct life span after setting endT
        cell1.endT = 500
        self.assertTrue(cell1.endT == 500)
        self.assertTrue(cell1.tau == 480)
        self.assertFalse(cell1.isUnfinished())

    def test_divide(self):
        """ Make sure cells divide properly with proper parent/child member variables. """
        