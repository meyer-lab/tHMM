""" This file is completely user defined. We have provided a general starting point for the user to use as an example. """
import scipy.stats as sp

class ObservationEmission:
    def __init__(self, bern_p, expon_scale_beta, gamma_a, gamma_scale): # user has to identify what parameters to use for each state
        """ Initialization function should take in just in the parameters for the observations that comprise the multivariate random variable emission they expect their data to have. """
        self.bern_p = bern_p
        self.expon_scale_beta = expon_scale_beta
        self.gamma_a = gamma_a
        self.gamma_scale = gamma_scale
        
    def rvs(self, size): # user has to identify what the multivariate (or univariate if he or she so chooses) random variable looks like
        """ User-defined way of calculating a random variable given the parameters of the state stored in that observation's object. """
        # {
        bern_obs = sp.bernoulli.rvs(p=self.bern_p, size=size) # bernoulli observations
        exp_obs = sp.expon.rvs(scale=self.expon_scale_beta, size=size) # exponential observations
        gamma_obs = sp.gamma.rvs(a=self.gamma_a, scale=self.gamma_scale, size=size) # gamma observations
        # } is user-defined in that they have to define and maintain the order of the multivariate random variables.
        # These tuples of observations will go into the cells in the lineage tree.
        return list(zip(bern_obs, exp_obs, gamma_obs))
    
    def pdf(self, tuple_of_obs): # user has to define how to calculate the likelihood
        """ User-defined way of calculating the likelihood of the observation stored in a cell. """
        # In the case of a univariate observation, the user still has to define how the likelihood is calculated,
        # but has the ability to just return the output of a known scipy.stats.<distribution>.<{pdf,pmf}> function.
        # In the case of a multivariate observation, the user has to decide how the likelihood is calculated.
        # In our example, we assume the observation's are uncorrelated across the dimensions (across the different
        # distribution observations), so the likelihood of observing the multivariate observation is just the product of 
        # the individual observation likelihoods.
        
        bern_ll = sp.bernoulli.pmf(k=tuple_of_obs[0], p=self.bern_p)  # bernoulli likelihood
        exp_ll = sp.expon.pdf(x=tuple_of_obs[1], scale=self.expon_scale_beta)  # exponential likelihood
        gamma_ll = sp.gamma.pdf(x=tuple_of_obs[2], a=self.gamma_a, scale=self.gamma_scale) # gamma likelihood
        
        return bern_ll*exp_ll*gamma_ll

    def estimator(self, list_of_tuples_of_obs):
        """ User-defined way of estimating the parameters given a list of the tuples of observations from a group of cells. """
        # unzipping the list of tuples
        unzipped_list_of_tuples_of_obs = list(zip(*list_of_tuples_of_obs))
        
        # getting the observations as individual lists
        bern_obs = list(unzipped_list_of_tuples_of_obs[0])
        exp_obs = list(unzipped_list_of_tuples_of_obs[1])
        gamma_obs = list(unzipped_list_of_tuples_of_obs[2])
        
        return result
        
# Because parameter estimation requires that estimators be written or imported, the user should be able to provide 
# estimators that can solve for the parameters that describe the distributions. We provide some estimators below as an example.
# Their use in the ObservationEmission class is shown in the estimator class method.

##--------------------- Estimating the Bernoulli Parameter ----------------------##

def bernoulli_estimator(bern_obs):
    """ Add up all the 1s and divide by the total length (finding the average). """
     return ( sum(bern_obs)+1e-10 )/( len(bern_obs)+2e-10 )  

##--------------------- Estimating the Exponential Parameter --------------------##

def exponential_estimator(exp_obs):
    """  """
    return ( sum(exp_obs) )/( len(exp_obs)+1 )

##------------------ Estimating the Gamma Distribution Parameters --------------------##

def gamma_estimator(gamma_obs):
    """
    An analytical estimator for two parameters of the Gamma distribution. Based on Thomas P. Minka, 2002 "Estimating a Gamma distribution".
    The likelihood function for Gamma distribution is:
    p(x | a, b) = Gamma(x; a, b) = x^(a-1)/(Gamma(a) * b^a) * exp(-x/b)
    Here we intend to find "a" and "b" given x as a sequence of data -- in this case
    the data is the cells' lifetime.
    To find the best estimate we find the value that maximizes the likelihood function.
    b_hat = x_bar / a
    using Newton's method to find the second parameter:
    a_hat =~ 0.5 / (log(x_bar) - log(x)_bar)
    Here x_bar means the average of x.
    Args:
        ----------
        gamma_obs (list): A list of gamma-distributed observations.
    Returns:
        ----------
        a_hat (float): The estimated value for shape parameter of the Gamma distribution
        b_hat (float): The estimated value for scale parameter of the Gamma distribution
    """
    tau1 = gamma_obs
    
    tau_mean = np.mean(tau1)
    tau_logmean = np.log(tau_mean)
    tau_meanlog = np.mean(np.log(tau1))

    # initialization step
    a_hat0 = 0.5 / (tau_logmean - tau_meanlog)  # shape
    psi_0 = np.log(a_hat0) - 1 / (2 * a_hat0)  # psi is the derivative of log of gamma function, which has been approximated as this term
    psi_prime0 = 1 / a_hat0 + 1 / (a_hat0 ** 2)  # this is the derivative of psi
    assert a_hat0 != 0, "the first parameter has been set to zero!"

    # updating the parameters
    for i in range(100):
        a_hat_new = (a_hat0 * (1 - a_hat0 * psi_prime0)) / (1 - a_hat0 * psi_prime0 + tau_meanlog - tau_logmean + np.log(a_hat0) - psi_0)
        b_hat_new = tau_mean / a_hat_new

        a_hat0 = a_hat_new
        psi_prime0 = 1 / a_hat0 + 1 / (a_hat0 ** 2)
        psi_0 = np.log(a_hat0) - 1 / (2 * a_hat0)
        psi_prime0 = 1 / a_hat0 + 1 / (a_hat0 ** 2)

        if np.abs(a_hat_new - a_hat0) <= 0.01:
            return [a_hat_new, b_hat_new]
        else:
            pass
    assert np.abs(a_hat_new - a_hat0) <= 0.01, "a_hat has not converged properly, a_hat_new - a_hat0 = {}".format(np.abs(a_hat_new - a_hat0))

    result = [a_hat_new, b_hat_new]
    return result
