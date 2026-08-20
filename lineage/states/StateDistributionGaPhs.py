"""State distribution class for separated G1 and G2 phase durations as observation."""

import numpy as np
from scipy.sparse import csr_array

from .stateCommon import censor_lineage_gaphs
from .StateDistributionGamma import StateDistribution as GammaSD


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
        self.params = np.array([bern_p1, bern_p2, gamma_a1, gamma_scale1, gamma_a2, gamma_scale2])
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

    def logpdf(self, x: np.ndarray) -> np.ndarray:
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

    def censor_lineage_array(
        self,
        censor_condition: int,
        tree: csr_array,
        obs: np.ndarray,
        states: np.ndarray,
        desired_experiment_time=2e12,
    ) -> tuple[csr_array, np.ndarray, np.ndarray]:
        """Applies censoring to array representation directly."""
        return censor_lineage_gaphs(tree, obs, states, censor_condition, desired_experiment_time)
