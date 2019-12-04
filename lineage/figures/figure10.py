"""
File: figure10.py
Purpose: Generates figure 10.
"""
import numpy as np
import matplotlib.pyplot as plt

from .figureCommon import getSetup


def makeFigure():
    """
    Makes figure 10.
    """

    ax, f = getSetup((21, 6), (1, 3))

    return f