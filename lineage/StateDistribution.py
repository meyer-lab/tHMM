""" This file is completely user defined. We have provided a general starting point for the user to use as an example. """
import numpy as np
import scipy.stats as sp
import math


class StateDistribution:
    def __init__(self, state, bern_p, gamma_a, gamma_scale):  # user has to identify what parameters to use for each state
        """ Initialization function should take in just in the parameters for the observations that comprise the multivariate random variable emission they expect their data to have. """
        self.state = state
        self.bern_p = bern_p
        self.gamma_a = gamma_a
        self.gamma_scale = gamma_scale

    def rvs(self, size):  # user has to identify what the multivariate (or univariate if he or she so chooses) random variable looks like
        """ User-defined way of calculating a random variable given the parameters of the state stored in that observation's object. It uses random variable generator functions of scipy.stats and makes a tuple out of them.
        Args:
        -----
        size {Int}: The desired number of random varibales for a specific observation.

        Returns:
        --------
        tuple_of_obs {list}: A list containing tuples of observations, for now it is (bernoulli for die/divide, exponential for lifetime, gamma for lifetime).
        """
        # {
        bern_obs = sp.bernoulli.rvs(p=self.bern_p, size=size)  # bernoulli observations
        gamma_obs = sp.gamma.rvs(a=self.gamma_a, scale=self.gamma_scale, size=size)  # gamma observations
        # } is user-defined in that they have to define and maintain the order of the multivariate random variables.
        # These tuples of observations will go into the cells in the lineage tree.
        tuple_of_obs = list(zip(bern_obs, gamma_obs))
        return tuple_of_obs

    def pdf(self, tuple_of_obs):  # user has to define how to calculate the likelihood
        """ User-defined way of calculating the likelihood of the observation stored in a cell. """
        # In the case of a univariate observation, the user still has to define how the likelihood is calculated,
        # but has the ability to just return the output of a known scipy.stats.<distribution>.<{pdf,pmf}> function.
        # In the case of a multivariate observation, the user has to decide how the likelihood is calculated.
        # In our example, we assume the observation's are uncorrelated across the dimensions (across the different
        # distribution observations), so the likelihood of observing the multivariate observation is just the product of
        # the individual observation likelihoods.

        bern_ll = sp.bernoulli.pmf(k=tuple_of_obs[0], p=self.bern_p)  # bernoulli likelihood
        gamma_ll = sp.gamma.pdf(x=tuple_of_obs[1], a=self.gamma_a, scale=self.gamma_scale)  # gamma likelihood

        return bern_ll * gamma_ll

    def estimator(self, list_of_tuples_of_obs):
        """ User-defined way of estimating the parameters given a list of the tuples of observations from a group of cells. It gathers the observations separately given the list of tuples, and passes them to the aforementioned function and finds the estimates for the parameters. Finally, returns them as a StateDistribution object with all the estimated parameters.
        Args:
        -----
        list_of_tuples_of_obs {list}: A list containing tuples of observations, for now it is (bernoulli for die/divide, exponential for lifetime, gamma for lifetime).

        Returns:
        --------
        state_estimate_obj {object}: A StateDistribution object instantiated with the estimated parameters.
        """
        # unzipping the list of tuples
        unzipped_list_of_tuples_of_obs = list(zip(*list_of_tuples_of_obs))

        # getting the observations as individual lists
        # {
        try:
            bern_obs = list(unzipped_list_of_tuples_of_obs[0])
            gamma_obs = list(unzipped_list_of_tuples_of_obs[1])
        except BaseException:
            # bernoulli observations
            bern_obs = [sp.bernoulli.rvs(p=0.9 * (np.random.uniform()))]
            gamma_obs = [sp.gamma.rvs(a=7.5 * (np.random.uniform()), scale=1.5 * (np.random.uniform()))]  # gamma observations


        bern_p_estimate = bernoulli_estimator(bern_obs)
        gamma_a_estimate, gamma_scale_estimate = gamma_estimator(gamma_obs)

        state_estimate_obj = StateDistribution(state=self.state,
                                               bern_p=bern_p_estimate,
                                               gamma_a=gamma_a_estimate,
                                               gamma_scale=gamma_scale_estimate)
        # } requires the user's attention.
        # Note that we return an instance of the StateDistribution class, but now instantiated with the parameters
        # from estimation. This is then stored in the original state distribution object which then gets updated
        # if this function runs again.
        return state_estimate_obj

    def __repr__(self):
        return "State object w/ parameters: {}, {}, {}.".format(self.bern_p, self.gamma_a, self.gamma_scale)



def prune_rule(cell):
    """ User-defined function that checks whether a cell's subtree should be removed. It takes in a cell, and checks its bernoulli observations, if the cell has died, then returns true. """
    truther = False
    if cell.obs[0] == 0:
        truther = True  # cell has died; subtree must be removed
    return truther


def tHMM_E_init(state):
    """ For every states, this function initiates an StateDistribution object with random arbitrary values for each parameter. This is used in the estimate class as the initial guess for parameter estimation."""
    return StateDistribution(state,
                             0.9 * (np.random.uniform()),
                             7.5 * (np.random.uniform()),
                             1.5 * (np.random.uniform()))

# Because parameter estimation requires that estimators be written or imported, the user should be able to provide
# estimators that can solve for the parameters that describe the distributions. We provide some estimators below as an example.
# Their use in the ObservationEmission class is shown in the estimator class method. User must take care to define estimators that
# can handle the case where the list of observations is empty.


def report_time(cell):
    """ Given any cell in the lineage, this helper function walks upward through the cell's ancestors and return how long it has taken from the beginning until now that this cell has been created. Ultimately, it is used to find out how long an experiment takes to create the lineage with the desired number of cells. """
    list_parents = [cell]
    taus = cell.obs[1]

    for cell in list_parents:
        if cell._isRootParent():
            break
        elif cell.parent not in list_parents:
            list_parents.append(cell.parent)
            taus += cell.parent.obs[1]
    return taus


def get_experiment_time(lineage):
    """ This function is to find the amount of time it took for the lineage to be created with the desired number of cells.  It applies the `report_time` function to all the leaf cells and finds out the tau for them, then return the longest tau amongst all of the leaf cells and reports it as the experiment time."""
    leaf_times = []
    for cell in lineage.output_leaves:
        temp = report_time(cell)
        leaf_times.append(temp)
    longest = max(leaf_times)
    return longest


def bernoulli_estimator(bern_obs):
    """ Add up all the 1s and divide by the total length (finding the average). """
    return (sum(bern_obs) + 1e-10) / (len(bern_obs) + 2e-10)


def exponential_estimator(exp_obs):
    """ Trivial exponential """
    return (sum(exp_obs) + 50e-10) / (len(exp_obs) + 1e-10)


# def gamma_estimator(gamma_obs):
#     """
#     An analytical estimator for two parameters of the Gamma distribution. Based on Thomas P. Minka, 2002 "Estimating a Gamma distribution".
#     The likelihood function for Gamma distribution is:
#     p(x | a, b) = Gamma(x; a, b) = x^(a-1)/(Gamma(a) * b^a) * exp(-x/b)
#     Here we intend to find "a" and "b" given x as a sequence of gamma distributed data.
#     To find the best estimate, we find the value that maximizes the likelihood of observing that data.
#     We fix b_hat as:

#     b_hat = x_bar / a

#     We then use Newton's method to find the second parameter:

#     a_hat ~= 0.5 / (log(x_bar) - (log(x))_bar)

#     Here x_bar means the average of x.
#     Args:
#     -----
#     gamma_obs {list}: A list of gamma-distributed random variables.

#     Returns:
#     --------
#     a_hat {float}: The estimated value for shape parameter of the Gamma distribution
#     b_hat {float}: The estimated value for scale parameter of the Gamma distribution
#     """
# #     b_hat = (np.var(gamma_obs) + 1e-7) / np.mean(gamma_obs)
# #     a_hat = (np.mean(gamma_obs) + 1e-7)/b_hat
# #     return a_hat, b_hat
#     tau1 = gamma_obs
#     tau_mean = np.mean(tau1)
#     tau_logmean = np.log(tau_mean)
#     tau_meanlog = np.mean(np.log(tau1))

#     # initialization step
#     a_hat0 = 0.5 / (tau_logmean - tau_meanlog)  # shape
#     psi_0 = np.log(a_hat0) - 1 / (2 * a_hat0)  # psi is the derivative of log of gamma function, which has been approximated as this term
#     psi_prime0 = 1 / a_hat0 + 1 / (2 * (a_hat0 ** 2))  # this is the derivative of psi
#     assert a_hat0 != 0, "the first parameter has been set to zero!"

#     # updating the parameters
#     for i in range(100):
#         a_hat_new = ((a_hat0 * (1 - a_hat0 * psi_prime0)) + 1e-6) / ((1 - a_hat0 * psi_prime0 + tau_meanlog - tau_logmean + np.log(a_hat0) - psi_0) + 1e-6)
#         assert math.isnan(a_hat_new) != True, "a_hat_new is nan"
#         b_hat_new = tau_mean / a_hat_new
#         assert math.isnan(b_hat_new) != True, "b_hat_new is nan"

#         a_hat0 = a_hat_new
#         psi_prime0 = 1 / a_hat0 + 1 / (a_hat0 ** 2)
#         psi_0 = np.log(a_hat0) - 1 / (2 * a_hat0)
#         psi_prime0 = 1 / a_hat0 + 1 / (a_hat0 ** 2)

#         if np.abs(a_hat_new - a_hat0) <= 0.01:
#             return a_hat_new, b_hat_new
#         else:
#             pass
#     assert np.abs(a_hat_new - a_hat0) <= 0.01, "a_hat has not converged properly, a_hat_new - a_hat0 = {}".format(np.abs(a_hat_new - a_hat0))

#     return a_hat_new, b_hat_new


def gamma_estimator(gamma_obs):
    """ This is a cloesd-form estimator for two parameters of the Gamma distribution, which is corrected for bias. """
    N = len(gamma_obs)
    a_val = 7.5
    b_val = 2.5
    if N > 1:
        x_lnx = [x * np.log(x) for x in gamma_obs]
        lnx = [np.log(x) for x in gamma_obs]
        # gamma_a
        a_val = (N * (sum(gamma_obs)))/(N * sum(x_lnx) - (sum(lnx)) * (sum(gamma_obs)))
        # gamma_scale
        b_val = (1/(N**2)) * (N * (sum(x_lnx)) - (sum(lnx)) * (sum(gamma_obs)))
        # bias correction
        a_val = (N /(N - 1)) * a_val
        # bias correction
        b_val = b_val - (1/N) * (3*b_val - (2/3) * (b_val/(b_val + 1)) - (4/5)* (b_val)/((1 + b_val)**2))

    return a_val, b_val


