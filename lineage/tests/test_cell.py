""" Unit test file. """
import unittest
import math
from ..CellNode import CellNode as c, generate

class TestModel(unittest.TestCase):
    """ Here are the unit tests. """

    def test_lifetime(self):
        """ Make sure cell lifetime variables behave properly. """
        cell1 = c(key=0, startT=20)

        # nan before setting death time
        self.assertTrue(math.isnan(cell1.endT))
        self.assertTrue(math.isnan(cell1.tau))
        self.assertTrue(cell1.isUnfinished())

        # correct life span after setting endT
        cell1.die(500)
        self.assertTrue(cell1.endT == 500)
        self.assertTrue(cell1.tau == 480)
        self.assertFalse(cell1.isUnfinished()) # cell is dead

    def test_divide(self):
        """ Make sure cells divide properly with proper parent/child member variables. """
        cell1 = c(key=0, startT=20)
        cell2, cell3 = cell1.divide(40)

        # cell divides at correct time & parent dies
        self.assertFalse(cell1.isUnfinished())
        self.assertTrue(cell1.tau == 20)
        self.assertTrue(cell2.startT == 40)
        self.assertTrue(cell2.isUnfinished())

        # left and right children exist for cell1 with proper linking
        self.assertTrue(cell1.left == cell2)
        self.assertTrue(cell1.right == cell3)
        self.assertTrue(cell2.parent == cell1)
        self.assertTrue(cell3.parent == cell1)
