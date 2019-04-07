import numpy as np
import pandas as pd

# read data into DataFrame

url1 = 'https://github.com/meyer-lab/lineage-growth/tree/master/lineage/G1_G2_duration_control'
df = pd.read_excel(url1)

##----------------------- Preprocessing the data ------------------------##

# dataFrmae into numpy array
a = df.values
G1 = a[:, 0]
G2 = a[:, 1]

# removing nan from the array
G2 = G2[~np.isnan(G2)]

# converting from unit of [frames] into [hours]
# every frame is every 30 minutes, so dividing the numbers by 2 gives unit of [hours]
G1 = G1/2
G2 = G2/2

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
            if p >= 0.01:
                print(dist[i], ':   ', 'p-value = ', p)
        p_val[val] = p

    return(p_val)

## --------------------- Check for our data ------------------------ ##
print('#### For G1 ####\n')
p_value = check_dist(G1, verbose = True)
print('\n #### For G2 ####\n')
p_value = check_dist(G2, verbose = True)

# What we get is:

#### probable distributions for G1: ####

# betaprime :    p-value =  0.9496245807703753
# gamma :    p-value =  0.7730477413285115
# erlang :    p-value =  0.7730478543522439

#### probable distributions for G2: ####

# betaprime :    p-value =  0.06029922688363665
# gamma :    p-value =  0.04329344124461376
# erlang :    p-value =  0.043293992146724136