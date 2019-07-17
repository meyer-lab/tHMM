''' This file contains all the emision distributions as objects that could be used easily, 
and we could add any new emission to it. '''
import scipy.stats as sp
from collections import OrderedDict
# importing functools for reduce() 
import functools 


class bernoulli:
    def __init__(self, param):
        self.param = param
    
    def rvs(self, size):
        return sp.bernoulli.rvs(self.param, size=size)
    
    def pmf(self, val):
        return sp.bernoulli.pmf(val, self.param)


class exponential:
    def __init__(self, param):
        self.param = param
        
    def rvs(self, size):
        return sp.expon.rvs(self.param, size=size)
    
    def pdf(self, val):
        return sp.expon.pdf(val, self.param)


class gamma:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
        
    def rvs(self, size)
        return sp.gamma.rvs(self.param1, scale=self.param2, size=size)

    def pdf(self, val):
        return sp.gamma.pdf(val, self.param1, scale = self.param2)

