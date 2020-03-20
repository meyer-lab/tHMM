""" This file contains the class for CellVar which holds the state and observation information in the hidden and observed trees respectively. """
import scipy.stats as sp
import numpy as np

# temporary style guide:
# Boolean functions are in camelCase.
# Functions that return cells or lists of cells will be spaced with underscores.
# Functions that are not to be used by a general user are prefixed with an underscore.
# States of cells are 0-indexed and discrete (states start at 0 and are whole numbers).
# Variables with the prefix num (i.e., num_states, num_cells) have an underscore following the 'num'.
# Docstrings use """ and not '''.
# Class names use camelCase.


class CellVar:
    """
    Cell class.
    """

    def __init__(self, state, parent, gen, **kwargs):
        """
        Instantiates the cell object.
        Contains memeber variables that identify daughter cells
        and parent cells. Also contains the state of the cell.
        """
        self.state = state
        self.parent = parent
        self.gen = gen
        self.censored = False

        if kwargs:
            self.left = kwargs.get("left", None)
            self.right = kwargs.get("right", None)
            self.obs = kwargs.get("obs", [])
            self.censored = kwargs.get("censored", True)

    def divide(self, T):
        """
        Member function that performs division of a cell.
        Equivalent to adding another timestep in a Markov process.
        """
        # roll a loaded die according to the row in the transtion matrix
        left_state, right_state = double(self.state, T)
        self.left = CellVar(state=left_state, parent=self, gen=self.gen + 1)
        self.right = CellVar(state=right_state, parent=self, gen=self.gen + 1)

        return self.left, self.right

    def isLeafBecauseTerminal(self):
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

    def isLeafBecauseDaughtersAreCensored(self):
        """
        Boolean.
        Returns true when a cell is a leaf because its children are censored.
        """
        if hasattr(self.left, "censored") and hasattr(self.right, "censored"):
            if self.left.censored and self.right.censored:
                return True

        return False

    def isLeaf(self):
        return self.isLeafBecauseTerminal() or self.isLeafBecauseDaughtersAreCensored()

    def isParent(self):
        """
        Boolean.
        Returns true if the cell has daughters.
        """
        return not self.isLeaf()

    def isChild(self):
        """
        Boolean.
        Returns true if this cell has a known parent.
        """
        if self.parent:
            return self.parent.isParent()

        return False

    def isRootParent(self):
        """
        Boolean.
        Returns true if this cell is the first cell in a lineage.
        """
        if not self.parent and self.gen == 1:
            return True

        return False

    def get_sister(self):
        """
        Member function that gets the sister of the current cell.
        """
        cell_to_return = None
        if self.parent.left is self:
            cell_to_return = self.parent.right
        elif self.parent.right is self:
            cell_to_return = self.parent.left
        return cell_to_return

    def get_root_cell(self):
        """
        Get the first cell in the lineage to which this cell belongs.
        """
        curr_cell = self
        while curr_cell.gen > 1:
            curr_cell = curr_cell.parent
        assert curr_cell.isRootParent()
        return curr_cell

    def get_daughters(self):
        """
        Get the left and right daughters of a cell if they exist.
        """
        temp = []
        if hasattr(self, "left") and hasattr(self, "right"):
            if self.left is not None and not self.left.censored:
                temp.append(self.left)
            if self.right is not None and not self.right.censored:
                temp.append(self.right)
        return temp

    def __repr__(self):
        """
        Printing function.
        """
        str_print = ""
        if hasattr(self, "obs"):
            str_print = "\n Generation: {}, State: {}, Observation: {}".format(self.gen, self.state, self.obs)
        else:
            str_print = "\n Generation: {}, State: {}, Observation: {}".format(self.gen, self.state, "This cell has no observations to report.")
        return str_print

    def __str__(self):
        return self.__repr__()


def double(parent_state, T):
    """
    Function that essentially rolls two of the same loaded dice
    given a state that determines the row of the transition matrix.
    The results of the roll of the loaded dice are two new states that are returned.
    """
    # Checking that the inputs are of the right shape
    assert (
        T.shape[0] == T.shape[1]
    ), "Transition numpy array is not square. Ensure that your transition numpy array has the same number of rows and columns."
    T_num_states = T.shape[0]
    assert 0 <= parent_state <= T_num_states - 1, "The parent state is a state outside of the range of states being considered."

    # Rolling two of the same loaded dice separate times and assigning
    # where they landed to states

    left_state_results, right_state_results = sp.multinomial.rvs(n=1, p=np.squeeze(T[parent_state, :]), size=2)
    left_state = left_state_results.tolist().index(1)
    right_state = right_state_results.tolist().index(1)

    return left_state, right_state
