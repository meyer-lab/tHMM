""" This file contains the class for CellVar which holds the state and observation information in the hidden and observed trees respectively. """
import numpy as np
from typing import TypeVar, Generic, Tuple
from dataclasses import dataclass


cellType = TypeVar('cellType')

class CellVar(Generic[cellType]):
    """
    Cell class.
    """

    def __init__(self, parent: cellType, gen: int, **kwargs):
        """Instantiates the cell object.
        Contains memeber variables that identify daughter cells
        and parent cells. Also contains the state of the cell.
        """
        self.parent = parent
        self.gen = gen
        self.observed = True

        if kwargs:
            self.state = kwargs.get("state", None)
            self.left = kwargs.get("left", None)
            self.right = kwargs.get("right", None)
            self.obs = kwargs.get("obs", [])

    def divide(self, T: np.ndarray) -> Tuple[cellType, cellType]:
        """
        Member function that performs division of a cell.
        Equivalent to adding another timestep in a Markov process.
        """
        # Checking that the inputs are of the right shape
        assert T.shape[0] == T.shape[1]

        # roll a loaded die according to the row in the transtion matrix
        left_state, right_state = np.random.choice(T.shape[0], size=2, p=T[self.state, :])
        self.left = CellVar(state=left_state, parent=self, gen=self.gen + 1)
        self.right = CellVar(state=right_state, parent=self, gen=self.gen + 1)

        return self.left, self.right

    def isLeafBecauseTerminal(self) -> bool:
        """
        Boolean.
        Returns true when a cell is a leaf with no children.
        These are cells at the end of the tree.
        """
        # if it has a left and right attribute able to be checked
        if hasattr(self, "left") and hasattr(self, "right"):
            # then check that they both DO not exist
            return self.left is None and self.right is None
        # otherwise, it has no left and right daughters
        return True

    def isLeafBecauseDaughtersAreNotObserved(self) -> bool:
        """
        Boolean.
        Returns true when a cell is a leaf because its children are unobserved
        but it itself is observed.
        """
        if hasattr(self.left, "observed") and hasattr(self.right, "observed"):
            # if its daughters are unobserved and it itself is observed
            if not self.left.observed and not self.right.observed and self.observed:
                return True
        # otherwise, it itself is observed and at least one of its daughters is observed
        return False

    def isLeaf(self) -> bool:
        """
        Boolean.
        Returns true when a cell is a leaf defined by the two conditions that determine
        whether a cell is a leaf. A cell only has to satisfy one of the conditions
        (an or statement) for it to be a leaf.
        """
        return self.isLeafBecauseTerminal() or self.isLeafBecauseDaughtersAreNotObserved()

    def isRootParent(self) -> bool:
        """
        Boolean.
        Returns true if this cell is the first cell in a lineage.
        """
        if not self.parent and self.gen == 1:
            return True

        return False

    def get_sister(self) -> cellType:
        """
        Member function that gets the sister of the current cell.
        """
        cell_to_return = None
        if self.parent.left is self:
            cell_to_return = self.parent.right
        elif self.parent.right is self:
            cell_to_return = self.parent.left
        return cell_to_return

    def get_root_cell(self) -> cellType:
        """
        Get the first cell in the lineage to which this cell belongs.
        """
        curr_cell = self
        while curr_cell.gen > 1:
            curr_cell = curr_cell.parent
        assert curr_cell.isRootParent()
        return curr_cell

    def get_daughters(self) -> list:
        """
        Get the left and right daughters of a cell if they exist.
        """
        temp = []
        if hasattr(self, "left") and hasattr(self, "right"):
            if self.left is not None and self.left.observed:
                temp.append(self.left)
            if self.right is not None and self.right.observed:
                temp.append(self.right)
        return temp


def tree_recursion(cell: cellType, subtree: list) -> None:
    """
    A recursive helper function that traverses upwards from the leaf to the root.
    """
    if cell.isLeaf():
        return
    subtree.append(cell.left)
    subtree.append(cell.right)
    tree_recursion(cell.left, subtree)
    tree_recursion(cell.right, subtree)
    return


def get_subtrees(node: cellType, lineage: list) -> Tuple[list, list]:
    """
    Given one cell, return the subtree of that cell,
    and return all the tree other than that subtree.
    """
    subtree = [node]
    tree_recursion(node, subtree)
    not_subtree = [cell for cell in lineage if cell not in subtree]
    return subtree, not_subtree


def find_two_subtrees(cell: cellType, lineage: list) -> Tuple[list, list, list]:
    """
    Gets the left and right subtrees from a cell.
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
