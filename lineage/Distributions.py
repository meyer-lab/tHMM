''' This file contains all the emision distributions as objects that could be used easily, and a user can add any new emissions to it. The idea is for the user to be able to define any emission or observation that follows a distribution and utilize it within the Markov model. The onus is on the user to define the emissions and distributions he or she wants to use, but we provide some distributions as examples. '''
import scipy.stats as sp

class bernoulli:
    def __init__(self, bern_p):
        self.bern_p = bern_p
    
    def rvs(self, size):
        return sp.bernoulli.rvs(p=self.bern_p, size=size)
    
    def pdf(self, bern_obs):
        return sp.bernoulli.pmf(k=bern_obs, p=self.bern_p)
    
    #def estimator

class expon:
    def __init__(self, expon_scale_beta):
        self.expon_scale_beta = expon_scale_beta
        
    def rvs(self, size):
        return sp.expon.rvs(self.expon_scale_beta, size=size)
    
    def pdf(self, expon_obs):
        return sp.expon.pdf(expon_obs, self.expon_scale_beta)

    #def estimator

class gamma:
    def __init__(self, gamma_a, gamma_scale):
        self.gamma_a = gamma_a
        self.gamma_scale = gamma_scale
        
    def rvs(self, size)
        return sp.gamma.rvs(a=self.gamma_a, scale=self.gamma_scale, size=size)

    def pdf(self, gamma_obs):
        return sp.gamma.pdf(gamma_obs, a=self.gamma_a, scale=self.gamma_scale)

    #def estimator
