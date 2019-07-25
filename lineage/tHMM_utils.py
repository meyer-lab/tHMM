'''Utility and helper functions for recursions and other needs in the tHMM class. This also contains the methods for AIC and accuracy.'''

import itertools
import numpy as np

def get_gen(gen, lineage):
    """
    Creates a list with all cells in the given generation
    Args:
        ----------
        gen (int): the generation number that we want to separate from the rest.
        lineage (list of objects): a list holding the objects (cells) in a lineage.

    Returns:
        ----------
        first_set (list of objects): a list that holds the cells with the same given
        generation.
    """
    first_set = []
    for cell in lineage:
        if cell.gen == gen:
            first_set.append(cell)
    return first_set

def get_parents_for_level(level, lineage):
    """
    Returns a set of all the parents of all the cells in a
    given level/generation. For example this would give you
    all the non-leaf cells in the generation above the one given.

    Args:
        ----------
        level (list of objects): a list that holds objects (cells) in a given level
        (or generation).
        lineage (list of objects): a list hodling objects (cells) in a lineage

    Returns:
        ----------
        parent_holder (set): a list that holds objects (cells) which
        are the parents of the cells in a given generation
    """
    parent_holder = set()  # set makes sure only one index is put in and no overlap
    for cell in level:
        parent_cell = cell.parent
        parent_holder.add(lineage.index(parent_cell))
    return parent_holder
