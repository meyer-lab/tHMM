""" This file is completely user defined. We have provided a general starting point for the user to use as an example. """
import numpy as np
import scipy as scip
import scipy.stats as sp


class StateDistribution:
    def __init__(self, state, bern_p, gamma_a, gamma_scale):  # user has to identify what parameters to use for each state
        """ Initialization function should take in just in the parameters for the observations that comprise the multivariate random variable emission they expect their data to have. """
        self.state = state
        self.bern_p = bern_p
        self.gamma_a = gamma_a
        self.gamma_scale = gamma_scale
        self.params = [self.bern_p, self.gamma_a, self.gamma_loc, self.gamma_scale]

    def rvs(self, size):  # user has to identify what the multivariate (or univariate if he or she so chooses) random variable looks like
        """ User-defined way of calculating a random variable given the parameters of the state stored in that observation's object. """
        # {
        bern_obs = sp.bernoulli.rvs(p=self.bern_p, size=size)  # bernoulli observations
        gamma_obs = sp.gamma.rvs(a=self.gamma_a, scale=self.gamma_scale, size=size)  # gamma observations
        # } is user-defined in that they have to define and maintain the order of the multivariate random variables.
        # These tuples of observations will go into the cells in the lineage tree.
        list_of_tuple_of_obs = list(zip(bern_obs, gamma_obs))
        return list_of_tuple_of_obs

    def pdf(self, tuple_of_obs):  # user has to define how to calculate the likelihood
        """ User-defined way of calculating the likelihood of the observation stored in a cell. """
        # In the case of a univariate observation, the user still has to define how the likelihood is calculated,
        # but has the ability to just return the output of a known scipy.stats.<distribution>.<{pdf,pmf}> function.
        # In the case of a multivariate observation, the user has to decide how the likelihood is calculated.
        # In our example, we assume the observation's are uncorrelated across the dimensions (across the different
        # distribution observations), so the likelihood of observing the multivariate observation is just the product of
        # the individual observation likelihoods.

        nPos = np.sum(np.array(tuple_of_obs[0]) == 1)
        nNeg = np.sum(np.array(tuple_of_obs[0]) == 0)
        bern_ll = np.prod(self.bern_p ** nPos * (1.0 - self.bern_p) ** nNeg)

        gxx = np.array(tuple_of_obs[1]) / self.gamma_scale
        gamma_ll = np.prod(np.power(gxx, self.gamma_a - 1.0) * np.exp(-gxx) / scip.special.gamma(self.gamma_a))

        return bern_ll * gamma_ll

    def estimator(self, list_of_tuples_of_obs):
        """ User-defined way of estimating the parameters given a list of the tuples of observations from a group of cells. """
        # unzipping the list of tuples
        unzipped_list_of_tuples_of_obs = list(zip(*list_of_tuples_of_obs))

        # getting the observations as individual lists
        # {
        try:
            bern_obs = list(unzipped_list_of_tuples_of_obs[0])
            gamma_obs = list(unzipped_list_of_tuples_of_obs[1])
        except BaseException:
            bern_obs = []
            gamma_obs = []

        bern_p_estimate = bernoulli_estimator(bern_obs)
        gamma_a_estimate, gamma_scale_estimate = gamma_estimator(gamma_obs)

        state_estimate_obj = StateDistribution(state=self.state, bern_p=bern_p_estimate, gamma_a=gamma_a_estimate, gamma_scale=gamma_scale_estimate)
        # } requires the user's attention.
        # Note that we return an instance of the state distribution class, but now instantiated with the parameters
        # from estimation. This is then stored in the original state distribution object which then gets updated
        # if this function runs again.
        return state_estimate_obj

    def __repr__(self):
        return "State object w/ parameters: {}, {}, {}, {}.".format(self.bern_p, self.gamma_a, self.gamma_scale)


def tHMM_E_init(state):
    return StateDistribution(state, 0.9, 10 * (np.random.uniform()), 1)


class Time:
    """
    Class that stores all the time related observations in a neater format.
    This will assist in pruning based on experimental time as well as
    obtaining attributes of the lineage as a whole, such as the
    average growth rate.
    """

    def __init__(self, startT, lifetime, endT):
        self.startT = startT
        self.lifetime = lifetime
        self.endT = endT  # equivalent to endT


def assign_times(lineageObj):
    """
    Assigns the start and end time for each cell in the lineage.
    The time observation will be stored in the cell's observation parameter list
    in the second position (index 1). See the other time functions to understand.
    """
    # traversing the cells by generation
    for gen_minus_1, level in enumerate(lineageObj.full_list_of_gens[1:]):
        true_gen = gen_minus_1 + 1  # generations are 1-indexed
        if true_gen == 1:
            for cell in level:
                assert cell._isRootParent()
                cell.time = Time(0, cell.obs[1], cell.obs[1])
        else:
            for cell in level:
                cell.time = Time(cell.parent.time.endT, cell.obs[1], cell.parent.time.endT + cell.obs[1])


def get_experiment_time(lineageObj):
    """
    This function returns the longest experiment time
    experienced by cells in the lineage.
    We can simply find the leaf cell with the
    longest end time. This is effectively
    the same as the experiment time for synthetic lineages.
    """
    longest = 0.0
    for cell in lineageObj.output_leaves:
        if cell.time.endT > longest:
            longest = cell.time.endT
    return longest


def track_lineage_generation_histogram(lineageObj):
    """
    This function creates list of lists (as many lists as states)
    that collects the number
    of cells in each state throughout the experiment based on
    successive generations.
    """
    max_gen = lineageObj.output_max_gen
    hist = np.zeros(shape=(lineageObj.num_states, max_gen))
    for gen_minus_1, level in enumerate(lineageObj.output_list_of_gens[1:]):
        true_gen = gen_minus_1 + 1  # generations are 1-indexed
        if true_gen == max_gen:
            assert gen_minus_1 == hist.shape[1] - 1
        for state in range(lineageObj.num_states):
            hist[state, gen_minus_1] = sum([1 if cell.state == state else 0 for cell in level])
    return hist


def track_population_generation_histogram(population):
    """
    This function runs the tracking function on a list of lineages.
    """
    collector = []
    for idx, lineageObj in enumerate(population):
        hist = track_lineage_generation_histogram(lineageObj)
        collector.append(hist)
    total = []
    for state in range(population[0].num_states):
        tmp_array = np.zeros(len(collector[0][0, :]))
        for idx, hist in enumerate(collector):
            if len(tmp_array) < len(hist[state, :]):
                c = hist[state, :].copy()
                c[: len(tmp_array)] += tmp_array
                tmp_array = c
            else:
                c = tmp_array.copy()
                c[: len(hist[state, :])] += hist[state, :]
                tmp_array = c
        total.append(tmp_array)

    return total


def track_lineage_growth_histogram(lineageObj, delta_time):
    """
    This function creates list of lists (as many lists as states)
    that divides the total experiment time
    into a certain number of bins (provided by the user) and collects the number
    of cells in each state throughout the experiment based on the time
    observations.
    """
    experiment_time = get_experiment_time(lineageObj)
    bins = int(np.ceil(experiment_time / delta_time))
    hist = np.zeros(shape=(lineageObj.num_states, bins))
    for state in range(lineageObj.num_states):
        start_time = 0
        end_time = start_time + delta_time
        for bin_idx in range(bins):
            num_alive = 0
            for cell_idx, cell in enumerate(lineageObj.output_lineage):
                if cell.state == state and cell.time.startT <= start_time and cell.time.endT >= end_time:
                    num_alive += 1
            start_time += delta_time
            end_time += delta_time
            hist[state, bin_idx] = num_alive
    return (hist, bins)


def track_population_growth_histogram(population, delta_time):
    """
    This function runs the tracking function on a list of lineages.
    """
    collector = []
    for idx, lineage in enumerate(population):
        hist, bins = track_lineage_growth_histogram(lineage, delta_time)
        collector.append(hist)
    total = []
    for state in range(population[0].num_states):
        tmp_array = np.zeros(len(collector[0][0, :]))
        for idx, hist in enumerate(collector):
            if len(tmp_array) < len(hist[state, :]):
                c = hist[state, :].copy()
                c[: len(tmp_array)] += tmp_array
                tmp_array = c
            else:
                c = tmp_array.copy()
                c[: len(hist[state, :])] += hist[state, :]
                tmp_array = c
        total.append(tmp_array)
    return total


def fate_prune_rule(cell):
    """
    User-defined function that checks whether a cell's subtree should be removed.
    Our example is based on the standard requirement that the first observation
    (index 0) is a measure of the cell's fate (1 being alive, 0 being dead).
    Clearly if a cell has died, its subtree must be removed.
    """
    return cell.obs[0] == 0


def time_prune_rule(cell, desired_experiment_time):
    """
    User-defined function that checks whether a cell's subtree should be removed.
    Our example is based on the standard requirement that the second observation
    (index 1) is a measure of the cell's lifetime.
    If a cell has lived beyond a certain experiment time, then its subtree
    must be removed.
    """
    return cell.time.endT > desired_experiment_time


# Because parameter estimation requires that estimators be written or imported,
# the user should be able to provide
# estimators that can solve for the parameters that describe the distributions.
# We provide some estimators below as an example.
# Their use in the StateDistribution class is shown in the estimator class method.
# User must take care to define estimators that
# can handle the case where the list of observations is empty.


def bernoulli_estimator(bern_obs):
    """ Add up all the 1s and divide by the total length (finding the average). """
    return (sum(bern_obs) + 1e-10) / (len(bern_obs) + 2e-10)


def gamma_estimator(gamma_obs):
    """ This is a closed-form estimator for two parameters of the Gamma distribution, which is corrected for bias. """
    N = len(gamma_obs)

    if N == 0:
        return 10, 1

    x_lnx = [x * np.log(x) for x in gamma_obs]
    lnx = [np.log(x) for x in gamma_obs]
    # gamma_a
    a_hat = (N * (sum(gamma_obs)) + 1e-10) / (N * sum(x_lnx) - (sum(lnx)) * (sum(gamma_obs)) + 1e-10)
    # gamma_scale
    b_hat = ((1 + 1e-10) / (N ** 2 + 1e-10)) * (N * (sum(x_lnx)) - (sum(lnx)) * (sum(gamma_obs)))

    if b_hat < 1.0 or 50.0 < a_hat < 5.0:
        return 10, 1

    return a_hat, b_hat
