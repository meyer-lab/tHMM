""" Unit test file. """
import unittest
import math
from ..CellNode import CellNode as c, generate

class TestModel(unittest.TestCase):
    """ Here are the unit tests. """

    def test_lifetime(self):
        """ Make sure the cell isUnfinished before the cell dies and then make sure the cell's lifetime (tau) is calculated properly after it dies. """
        cell1 = c(startT=20)

        # nan before setting death time
        self.assertTrue(math.isnan(cell1.tau))
        self.assertTrue(cell1.isUnfinished())

        # correct life span after setting endT
        cell1.die(500)
        self.assertTrue(cell1.tau == 480)
        self.assertFalse(cell1.isUnfinished()) # cell is dead

    def test_divide(self):
        """ Make sure cells divide properly with proper parent/child member variables. """
        cell1 = c(startT=20)
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

    def test_generate(self):
        """ Make sure we can generate fake data properly. """
        # if cell always divides it will stop at the maximum cell count when odd and one cell below when even (you can't divide and produce only 1 cell)
        out1 = generate(7, 1.0, 0.6)
        self.assertTrue(len(out1) == 7)
        out1 = generate(10, 1.0, 0.6)
        self.assertTrue(len(out1) == 9)

        # only 1 cell no matter numCells when cells always die
        out1 = generate(7, 0.0, 0.6)
        self.assertTrue(len(out1) == 1)

        # when locBern is 0.5 the initial cell divides ~1/2 the time
        nDiv = 0
        for i in range(1000):
            out1 = generate(3, 0.5, 0.6) # allow for 1 division max
            if len(out1) == 3:
                nDiv += 1
        self.assertTrue(450 <= nDiv <= 550) # assert that it divided ~500 times
