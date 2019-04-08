import scipy.stats as sp
import math


#######------Check the list of distributions and fit to the data --------#######

def check_dist(data, verbose=False):
    '''
    The function to check the data against distributions.

    This function uses fit method in scipy package to find the maximum likelihood estimate
    distribution that fits the data, p-value is then obtained from ks-test, and returns
    the best distribution and its p-value.

    Args:
    -----------
    data (1D-array): the data we want to find the fit distribution for.
    dist (list of str): a list of strings that are pre-defined distributions in scipy package
    verbose (bool): if it is True, when we call the function it will print out every distribution
    and its p-value that is evaluated.

    Returns:
    -----------
    p_val (dictionary): a dictionary containing the name of distributions with their corresponding p-value

    '''
    # A list of candidate distributions with [0, inf] range:
    dist = ['betaprime', 'fatiguelife', 'chi', 'expon', 'f', 'foldnorm',
            'frechet_r', 'frechet_l', 'gamma', 'invgamma', 'gompertz',
            'fisk', 'lognorm', 'loggamma', 'nakagami', 'pareto', 'rayleigh',
            'dweibull']

    p_val = {}
    for i, val in enumerate(dist):
        parameters = eval('sp.' + val + '.fit(data, fscale =1)')

        D, p = sp.kstest(data, val, args=parameters)

        if verbose:
            if p >= 0.01:
                print(dist[i], ':   ', 'p-value = ', p)
        p_val[val] = p

    return(p_val)