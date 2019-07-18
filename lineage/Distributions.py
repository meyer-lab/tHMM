""" This file is completely user defined"""
import scipy.stats as sp

class observation:
    def __init__(self, bern_p, expon_scale_beta, gamma_a, gamma_scale): # user has to identify what parameters to use for each state
        """ Initialization function should take in just in the parameters for the observations that comprise the multivariate random variable emission they expect their data to have. """
        self.bern_p = bern_p
        self.expon_scale_beta = expon_scale_beta
        self.gamma_a = gamma_a
        self.gamma_scale = gamma_scale
        
    def rvs(self, size): # user has to identify what the multivariate (or univariate if he or she so chooses) random variable looks like
        """ User defined way of calculating a random variable given the parameters of the state stored in that observation's object. """
        # {
        bern_obs = sp.bernoulli.rvs(p=self.bern_p, size=size)
        exp_obs = sp.expon.rvs(scale=self.expon_scale_beta, size=size)
        gamma_obs = sp.gamma.rvs(a=self.gamma_a, scale=self.gamma_scale, size=size)
        # } is user-defined in that they have to define and maintain the order of the multivariate random variables.
        # These tuples of observations will go into the cells in the lineage tree.
        return list(zip(bern_obs, exp_obs, gamma_obs))
    
    def pdf(self, tuple_of_obs): # user has to define how to calculate the likelihood
        """ User defined way of calculating the likelihood of the observation stored in a cell. """
        # In the case of a univariate observation, the user still has to define how the likelihood is calculated,
        # but has the ability to just return the output of a known scipy.stats.<distribution>.<{pdf,pmf}> function.
        # In the case of a multivariate observation, the user has to decide how the likelihood is calculated.
        # In our example, we assume the observation's are uncorrelated across the dimensions (across the different
        # distribution observations), so the likelihood of observing the multivariate observation is just the product of 
        # the individual observation likelihoods.

    
    #def estimator
