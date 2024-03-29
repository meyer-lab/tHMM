""" This file contains the class for CellVar which holds the state and observation information in the hidden and observed trees respectively. """

import numpy as np
from typing import Optional
from dataclasses import dataclass


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


class CellVar:
    """
    Cell class.
    """

    parent: Optional["CellVar"]
    gen: int
    observed: bool
    state: int
    obs: np.ndarray
    time: Optional[Time]
    left: Optional["CellVar"]
    right: Optional["CellVar"]

    def __init__(self, parent: Optional["CellVar"], state: int = -1):
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
        self.obs = np.empty(0, dtype=float)

    def divide(self, T: np.ndarray, rng=None):
        """
        Member function that performs division of a cell.
        Equivalent to adding another timestep in a Markov process.
        :param T: The array containing the likelihood of a cell switching states.
        """
        rng = np.random.default_rng(rng)
        # Checking that the inputs are of the right shape
        assert T.shape[0] == T.shape[1]

        # roll a loaded die according to the row in the transtion matrix
        left_state, right_state = rng.choice(T.shape[0], size=2, p=T[self.state, :])
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
