import scipy.stats as sp
import seaborn as ss; ss.set()
import math
import warnings
warnings.filterwarnings('ignore')




#######------Check the list of distributions and fit to the data --------#######

def check_dist(data, verbose = False):
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
    best_dist (str): the best fit distribution found by ks-test
    p_val[best_dist] (float): the p-value corresponding to the best fit distribution 
    
    '''
    ### A list of candidate distributions with [0, inf] range:
    dist = ['betaprime', 'fatiguelife', 'chi', 'expon', 'f', 'foldnorm',
                 'frechet_r', 'frechet_l', 'gamma', 'erlang', 'invgamma', 'gompertz',
                'fisk', 'lognorm', 'loggamma', 'nakagami', 'pareto', 'rice', 'rayleigh',
                'dweibull']
    
    p_val = {}
    for i , val in enumerate(dist):
        parameters = eval('sp.'+val+'.fit(data, fscale =1)')
    
        D, p = sp.kstest(data, val, args = parameters)
    
        if math.isnan(D): D = 0
        if math.isnan(p): p = 0
            

        if verbose:
            print(dist[i], ':   ', 'p-value = ', p)
        p_val[val] = p
    
    best_dist = max(p_val, key = p_val.get)
    
    return(best_dist, p_val[best_dist])

