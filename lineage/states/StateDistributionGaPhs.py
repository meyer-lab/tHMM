""" State distribution class for separated G1 and G2 phase durations as observation. """

import numpy as np

from .stateCommon import basic_censor
from .StateDistributionGamma import StateDistribution as GammaSD
from ..CellVar import Time, CellVar


class StateDistribution:
    """For G1 and G2 separated as observations."""

    def __init__(
        self,
        bern_p1: float = 0.9,
        bern_p2: float = 0.75,
        gamma_a1: float = 7.0,
        gamma_scale1: float = 3.0,
        gamma_a2: float = 14.0,
        gamma_scale2: float = 6.0,
    ):  # user has to identify what parameters to use for each state
        """Initialization function should take in just in the parameters for the observations that comprise the multivariate random variable emission they expect their data to have."""
        self.params = np.array(
            [bern_p1, bern_p2, gamma_a1, gamma_scale1, gamma_a2, gamma_scale2]
        )
        self.G1 = GammaSD(bern_p=bern_p1, gamma_a=gamma_a1, gamma_scale=gamma_scale1)
        self.G2 = GammaSD(bern_p=bern_p2, gamma_a=gamma_a2, gamma_scale=gamma_scale2)

    def rvs(
        self, size: int, rng=None
    ):  # user has to identify what the multivariate (or univariate if he or she so chooses) random variable looks like
        """User-defined way of calculating a random variable given the parameters of the state stored in that observation's object."""
        # {
        rng = np.random.default_rng(rng)
        bern_obsG1, gamma_obsG1, gamma_censor_obsG1 = self.G1.rvs(size, rng=rng)
        bern_obsG2, gamma_obsG2, gamma_censor_obsG2 = self.G2.rvs(size, rng=rng)
        # } is user-defined in that they have to define and maintain the order of the multivariate random variables.
        # These tuples of observations will go into the cells in the lineage tree.
        return (
            bern_obsG1,
            bern_obsG2,
            gamma_obsG1,
            gamma_obsG2,
            gamma_censor_obsG1,
            gamma_censor_obsG2,
        )

    def dist(self, other):
        """Calculate the Wasserstein distance between this state emissions and the given."""
        assert isinstance(self, type(other))
        return self.G1.dist(other.G1) + self.G2.dist(other.G2)

    def dof(self):
        """Return the degrees of freedom."""
        return self.G1.dof() + self.G2.dof()

    def logpdf(self, x: np.ndarray):
        """To calculate the log-likelihood of observations to states."""

        G1_LL = self.G1.logpdf(x[:, np.array([0, 2, 4])])
        G2_LL = self.G2.logpdf(x[:, np.array([1, 3, 5])])

        return G1_LL + G2_LL

    def estimator(self, x: np.ndarray, gammas: np.ndarray):
        """User-defined way of estimating the parameters given a list of the tuples of observations from a group of cells."""
        self.G1.estimator(x[:, np.array([0, 2, 4])], gammas)
        self.G2.estimator(x[:, np.array([1, 3, 5])], gammas)

        self.params[0] = self.G1.params[0]
        self.params[1] = self.G2.params[0]
        self.params[2:4] = self.G1.params[1:3]
        self.params[4:6] = self.G2.params[1:3]

        # } requires the user's attention.
        # Note that we return an instance of the state distribution class, but now instantiated with the parameters
        # from estimation. This is then stored in the original state distribution object which then gets updated
        # if this function runs again.

    def censor_lineage(
        self,
        censor_condition: int,
        full_lineage: list[CellVar],
        desired_experiment_time=2e12,
    ):
        """
        This function removes those cells that are intended to be remove
        from the output binary tree based on emissions.
        It takes in LineageTree object, walks through all the cells in the output binary tree,
        applies the pruning to each cell that is supposed to be removed,
        and returns the censored list of cells.
        """
        # Assign times
        # traversing the cells by generation
        for ii, cell in enumerate(full_lineage):
            if ii == 0:
                cell.time = Time(0, cell.obs[2] + cell.obs[3])
                cell.time.transition_time = 0 + cell.obs[2]
            else:
                cell.time = Time(
                    cell.parent.time.endT,
                    cell.parent.time.endT + cell.obs[2] + cell.obs[3],
                )
                cell.time.transition_time = cell.parent.time.endT + cell.obs[2]

        if censor_condition == 0:
            return full_lineage

        for cell in full_lineage:
            if censor_condition in (1, 3):
                fate_censor(cell)

            if censor_condition in (2, 3):
                time_censor(cell, desired_experiment_time)

        basic_censor(full_lineage)

        return [c for c in full_lineage if c.observed]


def fate_censor(cell):
    """
    User-defined function that checks whether a cell's subtree should be removed.
    Our example is based on the standard requirement that the first observation
    (index 0) is a measure of the cell's fate (1 being alive, 0 being dead).
    Clearly if a cell has died, its subtree must be removed.
    """
    if cell.obs[0] == 0 or cell.obs[1] == 0:
        if not cell.isLeafBecauseTerminal():
            cell.left.observed = False
            cell.right.observed = False
        if cell.obs[0] == 0:  # dies in G1
            cell.obs[1] = float("nan")  # unobserved
            cell.obs[3] = float("nan")  # unobserved
            cell.obs[5] = float("nan")  # unobserved
            cell.time.endT = cell.time.startT + cell.obs[2]
            cell.time.transition_time = cell.time.endT
        if cell.obs[1] == 0:  # dies in G2
            cell.time.endT = cell.time.startT + cell.obs[2] + cell.obs[3]


def time_censor(cell, desired_experiment_time):
    """
    User-defined function that checks whether a cell's subtree should be removed.
    Our example is based on the standard requirement that the second observation
    (index 1) is a measure of the cell's lifetime.
    If a cell has lived beyond a certain experiment time, then its subtree
    must be removed.
    """
    if cell.time.endT > desired_experiment_time:
        cell.time.endT = desired_experiment_time
        cell.obs[1] = float("nan")  # unobserved
        cell.obs[3] = desired_experiment_time - cell.time.transition_time
        cell.obs[5] = 0  # censored
        if not cell.isLeafBecauseTerminal():
            cell.left.observed = False
            cell.right.observed = False

    if cell.time.transition_time > desired_experiment_time:
        cell.time.endT = desired_experiment_time
        cell.time.transition_time = desired_experiment_time
        cell.obs[0] = float("nan")  # unobserved
        cell.obs[1] = float("nan")  # unobserved
        cell.obs[2] = desired_experiment_time - cell.time.startT
        cell.obs[3] = float("nan")  # unobserved
        cell.obs[4] = 0  # censored
        cell.obs[5] = float("nan")  # unobserved
        if not cell.isLeafBecauseTerminal():
            cell.left.observed = False
            cell.right.observed = False
