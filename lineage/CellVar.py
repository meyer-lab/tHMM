""" This file contains the class for CellVar which holds the state and observation information in the hidden and observed trees respectively. """
from __future__ import annotations
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


class CellVar:
    """
    Cell class.
    """
    parent: Optional['CellVar']
    gen: int
    observed: bool
    state: Optional[int]
    obs: Optional[np.ndarray]
    left: Optional['CellVar']
    right: Optional['CellVar']

    def __init__(self, parent: Optional['CellVar'], state: Optional[int] = None):
        """Instantiates the cell object.
        Contains memeber variables that identify daughter cells
        and parent cells. Also contains the state of the cell.
        """
        self.parent = parent

        if parent is None:
            self.gen = 1
        else:
            self.gen = parent.gen + 1

        self.observed = True
        self.state = state
        self.left = None
        self.right = None
        self.obs = None

    def divide(self, T: np.ndarray):
        """
        Member function that performs division of a cell.
        Equivalent to adding another timestep in a Markov process.
        :param T: The array containing the likelihood of a cell switching states.
        """
        # Checking that the inputs are of the right shape
        assert T.shape[0] == T.shape[1]

        # roll a loaded die according to the row in the transtion matrix
        left_state, right_state = np.random.choice(T.shape[0], size=2, p=T[self.state, :])
        self.left = CellVar(state=left_state, parent=self)
        self.right = CellVar(state=right_state, parent=self)

        return self.left, self.right

    def isLeafBecauseTerminal(self) -> bool:
        """
        Returns true when a cell is a leaf with no children.
        These are cells at the end of the tree.
        """
        return self.left is None and self.right is None

    def isLeaf(self) -> bool:
        """
        Returns true when a cell is a leaf defined by the two conditions that determine
        whether a cell is a leaf. A cell only has to satisfy one of the conditions
        (an or statement) for it to be a leaf.
        """
        if self.isLeafBecauseTerminal():
            return True

        # Returns true when a cell is a leaf because its children are unobserved
        # but it itself is observed.
        if not self.left.observed and not self.right.observed and self.observed:  # type: ignore
            return True

        # otherwise, it itself is observed and at least one of its daughters is observed
        return False

    def isRootParent(self) -> bool:
        """
        Returns true if this cell is the first cell in a lineage.
        """
        return self.parent is None

    def get_sister(self):
        """
        :Return cell_to_return: The sister of the current cell.
        """
        cell_to_return = None
        if self.parent.left is self:
            cell_to_return = self.parent.right
        elif self.parent.right is self:
            cell_to_return = self.parent.left
        return cell_to_return

    def get_root_cell(self):
        """
        :Return curr_cell: The first cell in the lineage to which this cell belongs.
        """
        curr_cell = self
        while curr_cell.gen > 1:
            curr_cell = curr_cell.parent
        assert curr_cell.isRootParent()
        return curr_cell

    def get_daughters(self) -> list:
        """
        Get the left and right daughters of a cell if they exist.
        :return Temp: The list of existing daughter cells.
        """
        temp: list[CellVar] = list()
        if self.left is not None and self.left.observed:
            temp.append(self.left)
        if self.right is not None and self.right.observed:
            temp.append(self.right)
        return temp


def tree_recursion(cell, subtree: list) -> None:
    """
    A recursive helper function that traverses upwards from the leaf to the root.
    :param cell: An instantiation of the Cell class.
    :param subtree: A list of previous cells in the branch of a given cell.
    """
    if cell.isLeaf():
        return
    subtree.append(cell.left)
    subtree.append(cell.right)
    tree_recursion(cell.left, subtree)
    tree_recursion(cell.right, subtree)
    return


def get_subtrees(node, lineage: list) -> Tuple[list, list]:
    """
    Given one cell, return the subtree of that cell,
    and return all the tree other than that subtree.
    :param node: The location of the cell whose subtree will be returned.
    :param lineage: The list of cells originating from a specific daughter cell.
    :return subtree: A list of previous cells in the branch of a given cell.
    :return not_subtree: The list of cells that do not contain the subtree.
    """
    subtree = [node]
    tree_recursion(node, subtree)
    not_subtree = [cell for cell in lineage if cell not in subtree]
    return subtree, not_subtree


def find_two_subtrees(cell, lineage: list) -> Tuple[Optional[list], Optional[list], list]:
    """
    Gets the left and right subtrees from a cell.
    :param cell: An instantiation of the Cell class.
    :param lineage: The list of cells originating from a specific daughter cell.
    :return left_sub: The left subtree branching from a given cell.
    :param right_sub: The right subtree branching from a given cell.
    :param neither_subtree: The subtrees that are not branching from the given cell.
    """
    if cell.isLeaf():
        return None, None, lineage
    left_sub, _ = get_subtrees(cell.left, lineage)
    right_sub, _ = get_subtrees(cell.right, lineage)
    neither_subtree = [node for node in lineage if node not in left_sub and node not in right_sub]
    return left_sub, right_sub, neither_subtree


@dataclass(init=True, repr=True, eq=True, order=True)
class Time:
    """
    Class that stores all the time related observations in a neater format.
    This assists in pruning based on experimental time and obtaining
    attributes of the lineage as a whole like the average growth rate.
    """
    startT: float
    endT: float
    transition_time: float = 0.0
