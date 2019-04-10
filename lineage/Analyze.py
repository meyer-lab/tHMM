'''Calls the tHMM functions and outputs the parameters needed to generate the Figures'''

from .BaumWelch import fit
from .Viterbi import get_leaf_deltas, get_nonleaf_deltas, Viterbi
from .UpwardRecursion import get_leaf_Normalizing_Factors, get_leaf_betas, get_nonleaf_NF_and_betas, calculate_log_likelihood
from .tHMM import tHMM
from .tHMM_utils import getAccuracy


def Analyze(X, numStates, keepBern=True):
    '''Runs a tHMM and outputs state classification from viterbi, thmm object, normalizing factor, log likelihood, and deltas'''
    run = True
    while run:
        tHMMobj = tHMM(X, numStates=numStates, FOM='G', keepBern=keepBern) # build the tHMM class with X
        fit(tHMMobj, max_iter=200, verbose=True)
        if tHMMobj.paramlist[0]["E"][0, 1] < 1000 and tHMMobj.paramlist[0]["E"][1, 1] < 1000:
            run = False
    deltas, state_ptrs = get_leaf_deltas(tHMMobj)  # gets the deltas matrix
    get_nonleaf_deltas(tHMMobj, deltas, state_ptrs)
    all_states = Viterbi(tHMMobj, deltas, state_ptrs)
    NF = get_leaf_Normalizing_Factors(tHMMobj)
    betas = get_leaf_betas(tHMMobj, NF)
    get_nonleaf_NF_and_betas(tHMMobj, NF, betas)
    LL = calculate_log_likelihood(tHMMobj, NF)
    getAccuracy(tHMMobj, all_states, verbose=False)
    return(deltas, state_ptrs, all_states, tHMMobj, NF, LL)
