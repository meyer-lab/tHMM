""" This file is completely user defined. We have provided a general starting point for the user to use as an example. """

import numpy as np
import scipy.stats as sp
from typing import Union

from .stateCommon import gamma_estimator, basic_censor, bern_estimator
from ..CellVar import Time, CellVar


class StateDistribution:
    """
    StateDistribution for cells with gamma distributed times.
    """

    def __init__(
        self, bern_p: float = 0.9, gamma_a: float = 7.0, gamma_scale: float = 4.5
    ):
        """Initialization function should take in just in the parameters
        for the observations that comprise the multivariate random variable emission they expect their data to have.
        In this case, we used Gamma distribution for cell lifetime, which has 2 parameters; shape and scale.
        And we used bernoulli distribution for cell lifetime, which has 1 parameter.
        """
        self.params = np.array([bern_p, gamma_a, gamma_scale])

    def rvs(
        self, size: int, rng=None
    ):  # user has to identify what the multivariate (or univariate) random variable looks like
        """User-defined way of calculating a random variable given the parameters of the state stored in their object."""
        # {
        rng = np.random.default_rng(rng)
        bern_obs = rng.binomial(
            1, p=self.params[0], size=size
        )  # bernoulli observations
        gamma_obs = rng.gamma(
            self.params[1], scale=self.params[2], size=size
        )  # gamma observations
        gamma_obs_censor = [1] * size  # 1 if observed

        # } is user-defined in that they have to define and maintain the order of the multivariate random variables.
        # These tuples of observations will go into the cells in the lineage tree.
        return bern_obs, gamma_obs, gamma_obs_censor

    def dist(self, other) -> float:
        """Calculate the Wasserstein distance between two gamma distributions that each correspond to a state.
        This is our way of calculating the distance between two state, when their bernoulli distribution is kept the same.
        For more information about wasserstein distance, please see https://en.wikipedia.org/wiki/Wasserstein_metric.
        """
        assert isinstance(self, type(other))
        dist = np.absolute(
            self.params[1] * self.params[2] - other.params[1] * other.params[2]
        )
        return dist

    def dof(self) -> int:
        """Return the degrees of freedom.
        In this case, each state has 1 bernoulli distribution parameter, and 2 gamma distribution parameters.
        """
        return 3

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """User-defined way of calculating the log likelihood of the observation stored in a cell.
        In the case of a univariate observation, the user still has to define how the likelihood is calculated,
        but has the ability to just return the output of a known scipy.stats.<distribution>.<{pdf,pmf}> function.
        In the case of a multivariate observation, the user has to decide how the likelihood is calculated.
        In our example, we assume the observation's are uncorrelated across the dimensions (across the different
        distribution observations), so the total log likelihood of observing the multivariate observation is just the sum of
        the individual observation log likelihoods.
        """
        ll = np.zeros(x.shape[0])

        # Update uncensored Gamma
        ll[x[:, 2] == 1] += sp.gamma.logpdf(
            x[x[:, 2] == 1, 1], a=self.params[1], scale=self.params[2]
        )

        # Update censored Gamma
        ll[x[:, 2] == 0] += sp.gamma.logsf(
            x[x[:, 2] == 0, 1], a=self.params[1], scale=self.params[2]
        )

        # Remove dead cells
        ll[x[:, 0] == 0] = 0.0

        # Update for observed Bernoulli
        ll[np.isfinite(x[:, 0])] += sp.bernoulli.logpmf(
            x[np.isfinite(x[:, 0]), 0], self.params[0]
        )

        # Log likelihood of negative values should be zero
        ll[x[:, 1] < 0] = 0.0
        ll[x[:, 0] < 0] = 0.0

        return ll

    def estimator(self, X: list, gammas: np.ndarray):
        """User-defined way of estimating the parameters given a list of the tuples of observations from a group of cells."""

        # getting the observations as individual lists
        # {
        x = np.array(X)
        bern_obs = x[:, 0]
        γ_obs = x[:, 1]
        gamma_obs_censor = x[:, 2]

        # remove negative observations from fitting
        bern_obs_ = bern_obs[γ_obs >= 0]
        γ_obs_ = γ_obs[γ_obs >= 0]
        gamma_obs_censor_ = gamma_obs_censor[γ_obs >= 0]
        gammas_ = gammas[γ_obs >= 0]

        # Both unoberved and dead cells should be removed from gamma
        g_mask = np.logical_and(np.isfinite(γ_obs_), bern_obs_.astype("bool"))
        assert (
            np.sum(g_mask) > 0
        ), "All the cells are eliminated from the Gamma estimator."

        self.params[0] = bern_estimator(bern_obs, gammas)
        self.params[1], self.params[2] = gamma_estimator(
            [γ_obs_[g_mask]],
            [gamma_obs_censor_[g_mask]],
            [gammas_[g_mask]],
            self.params[1:3],
            phase="all",
        )

        # } requires the user's attention.
        # Note that we return an instance of the state distribution class, but now instantiated with the parameters
        # from estimation. This is then stored in the original state distribution object which then gets updated
        # if this function runs again.

    def assign_times(self, full_lineage: list[CellVar]):
        """
        Assigns the start and end time for each cell in the lineage.
        The time observation will be stored in the cell's observation parameter list
        in the second position (index 1). See the other time functions to understand.
        This is used in the creation of LineageTrees.
        """
        # traversing the cells by generation
        for ii, cell in enumerate(full_lineage):
            if ii == 0:
                cell.time = Time(0, cell.obs[1])
            else:
                cell.time = Time(
                    cell.parent.time.endT, cell.parent.time.endT + cell.obs[1]
                )

    def censor_lineage(
        self,
        censor_condition: int,
        full_lineage: list[CellVar],
        desired_experiment_time=2e12,
    ):
        """
        This function removes those cells that are intended to be removed.
        These cells include the descendants of a cell that has died, or has lived beyonf the experimental end time.
        It takes in LineageTree object, walks through all the cells in the output binary tree,
        applies the censorship to each cell that is supposed to be removed,
        and returns the lineage of cells that are supposed to be alive and accounted for.
        """
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
    Checks whether a cell has died based on its fate, and if so, it will remove its subtree.
    Our example is based on the standard requirement that the first observation
    (index 0) is a measure of the cell's fate (1 being alive, 0 being dead).
    """
    if cell.obs[0] == 0:
        if not cell.isLeafBecauseTerminal():
            cell.left.observed = False
            cell.right.observed = False


def time_censor(cell, desired_experiment_time: Union[int, float]):
    """
    Checks whether a cell has lived beyond the experiment end time and if so, it will remove its subtree.
    Our example is based on the standard requirement that the second observation
    (index 1) is a measure of the cell's lifetime.
    """
    if cell.time.endT > desired_experiment_time:
        cell.time.endT = desired_experiment_time
        cell.obs[0] = float("nan")
        cell.obs[1] = desired_experiment_time - cell.time.startT
        cell.obs[2] = 0  # censored
        if not cell.isLeafBecauseTerminal():
            # the daughters are no longer observed
            cell.left.observed = False
            cell.right.observed = False


def atonce_estimator(
    all_tHMMobj: list, x_list: list, gammas_list: list, phase: str, state_j: int
):
    """Estimating the parameters for one state, in this case bernoulli nad gamma distirbution parameters,
    given a list of the tuples of observations from a group of cells.
    gammas_list is only for one state."""
    # unzipping the list of tuples
    x_data = [np.array(x) for x in x_list]

    # getting the observations as individual lists
    bern_obs = [x[:, 0] for x in x_data]
    γ_obs = [x[:, 1] for x in x_data]
    gamma_obs_censor = [x[:, 2] for x in x_data]

    # remove negative observations from being used for fitting
    bern_obs_ = [b[γ_obs[i] >= 0] for i, b in enumerate(bern_obs)]
    γ_obs_ = [g[g >= 0] for i, g in enumerate(γ_obs)]
    gamma_obs_censor_ = [gc[γ_obs[i] >= 0] for i, gc in enumerate(gamma_obs_censor)]
    gammas_list_ = [gc[γ_obs[i] >= 0] for i, gc in enumerate(gammas_list)]

    for i, item in enumerate(bern_obs_):
        assert item.shape == γ_obs_[i].shape == gamma_obs_censor_[i].shape

    bern_params = [
        bern_estimator(bern_obs_[i], gammas_list_[i]) for i in range(len(gammas_list_))
    ]

    # Both unoberved and dead cells should be removed from gamma
    g_masks = [
        np.logical_and(np.isfinite(γ_o), berns.astype("bool"))
        for γ_o, berns in zip(γ_obs_, bern_obs_)
    ]
    for g_mask in g_masks:
        assert (
            np.sum(g_mask) > 0
        ), "All the cells are eliminated from the Gamma estimator."

    γ_obs_total = [g_obs[g_masks[i]] for i, g_obs in enumerate(γ_obs_)]
    γ_obs_total_censored = [
        g_obs_cen[g_masks[i]] for i, g_obs_cen in enumerate(gamma_obs_censor_)
    ]
    gammas_total = [
        np.vstack(gamma_tot)[g_masks[i]] for i, gamma_tot in enumerate(gammas_list_)
    ]
    gammas_total = [np.squeeze(g) for g in gammas_total]

    if phase == "G1":
        x0 = np.array(
            [all_tHMMobj[0].estimate.E[state_j].params[2]]
            + [tHMMobj.estimate.E[state_j].params[3] for tHMMobj in all_tHMMobj]
        )
        output = gamma_estimator(
            γ_obs_total, γ_obs_total_censored, gammas_total, x0, phase=phase
        )
        for i, tHMMobj in enumerate(all_tHMMobj):
            tHMMobj.estimate.E[state_j].params[0] = bern_params[i]
            tHMMobj.estimate.E[state_j].G1.params[0] = bern_params[i]
            tHMMobj.estimate.E[state_j].params[2] = output[0]
            tHMMobj.estimate.E[state_j].G1.params[1] = output[0]
            tHMMobj.estimate.E[state_j].params[3] = output[i + 1]
            tHMMobj.estimate.E[state_j].G1.params[2] = output[i + 1]

    elif phase == "G2":
        x0 = np.array(
            [all_tHMMobj[0].estimate.E[state_j].params[4]]
            + [tHMMobj.estimate.E[state_j].params[5] for tHMMobj in all_tHMMobj]
        )
        output = gamma_estimator(
            γ_obs_total, γ_obs_total_censored, gammas_total, x0, phase=phase
        )
        for i, tHMMobj in enumerate(all_tHMMobj):
            tHMMobj.estimate.E[state_j].params[1] = bern_params[i]
            tHMMobj.estimate.E[state_j].G2.params[0] = bern_params[i]
            tHMMobj.estimate.E[state_j].params[4] = output[0]
            tHMMobj.estimate.E[state_j].G2.params[1] = output[0]
            tHMMobj.estimate.E[state_j].params[5] = output[i + 1]
            tHMMobj.estimate.E[state_j].G2.params[2] = output[i + 1]

    elif phase == "all":
        x0 = np.array(
            [all_tHMMobj[0].estimate.E[state_j].params[1]]
            + [tHMMobj.estimate.E[state_j].params[2] for tHMMobj in all_tHMMobj]
        )
        output = gamma_estimator(
            γ_obs_total, γ_obs_total_censored, gammas_total, x0, phase=phase
        )
        for i, tHMMobj in enumerate(all_tHMMobj):
            tHMMobj.estimate.E[state_j].params[0] = bern_params[i]
            tHMMobj.estimate.E[state_j].params[1] = output[0]
            tHMMobj.estimate.E[state_j].params[2] = output[i + 1]
