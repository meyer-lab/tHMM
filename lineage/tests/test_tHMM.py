""" Unit test file. """
import unittest
import math
import numpy as np
from ..Lineage import Population as p, generatePopulationWithTime as gpt
from ..CellNode import CellNode as c, generateLineageWithTime, doublingTime

class TestModel(unittest.TestCase):
    """ Here are the unit tests. """
