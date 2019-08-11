'''Calls the tHMM functions and outputs the parameters needed to generate the Figures'''

from .BaumWelch import fit
from .Viterbi import get_leaf_deltas, get_nonleaf_deltas, Viterbi
from .UpwardRecursion import get_leaf_Normalizing_Factors, get_leaf_betas, get_nonleaf_NF_and_betas, calculate_log_likelihood
from .tHMM import tHMM
from .LineageTree import LineageTree
from .StateDistribution import StateDistribution


def Analyze(X, numStates):
    """Runs a tHMM and outputs state classification from viterbi, thmm object, normalizing factor, log likelihood, and deltas.
    Args:
    -----
    X {list}: A list containing LineageTree objects as lineages. 
    numStates {Int}: The number of states we want our model to estimate for the given population. 

    Returns:
    --------
    deltas {}:
    state_ptrs {}:
    all_states {}:
    tHMMobj {obj}:
    NF {}:
    LL {}:
    """

    tHMMobj = tHMM(X, numStates=numStates)  # build the tHMM class with X
    fit(tHMMobj, max_iter=200, verbose=True)

    deltas, state_ptrs = get_leaf_deltas(tHMMobj)  # gets the deltas matrix
    get_nonleaf_deltas(tHMMobj, deltas, state_ptrs)
    all_states = Viterbi(tHMMobj, deltas, state_ptrs)
    NF = get_leaf_Normalizing_Factors(tHMMobj)
    betas = get_leaf_betas(tHMMobj, NF)
    get_nonleaf_NF_and_betas(tHMMobj, NF, betas)
    LL = calculate_log_likelihood(tHMMobj, NF)
    return(deltas, state_ptrs, all_states, tHMMobj, NF, LL)

##-------------------- Figure 6 
def accuracy_increased_cells():
    """ Calclates accuracy and parameter estimation by increasing the number of cells in a lineage for a two-state model. """

    # pi: the initial probability vector
    pi = np.array([0.5, 0.5], dtype="float")

    # T: transition probability matrix
    T = np.array([[0.99, 0.01],
              [0.15, 0.85]])

    # State 0 parameters "Resistant"
    state0 = 0
    bern_p0 = 0.95
    expon_scale_beta0 = 80
    gamma_a0 = 5.0
    gamma_scale0 = 1.0

    # State 1 parameters "Susciptible"
    state1 = 1
    bern_p1 = 0.8
    expon_scale_beta1 = 40
    gamma_a1 = 10.0
    gamma_scale1 = 2.0

    state_obj0 = StateDistribution(state0, bern_p0, expon_scale_beta0, gamma_a0, gamma_scale0)
    state_obj1 = StateDistribution(state1, bern_p1, expon_scale_beta1, gamma_a1, gamma_scale1)

    E = [state_obj0, state_obj1]
    # the key part in this function
    desired_num_cells = [2**5 - 1, 2**7 - 1, 2**8 - 1, 2**9 - 1, 2**10 - 1, 2**11 - 1]
    
    for num in desired_num_cells:
        # unpruned lineage
        lineage_unpruned = LineageTree(pi, T, E, num, prune_boolean=False)
        # pruned lineage
        lineage_pruned = LineageTree(pi, T, E, num, prune_boolean=True)

        X1 = [lineage1]
        X2 = [lineage2]
        

    
    
    
##-------------------- Figure 7   
def accuracy_increased_lineages():
    """ Calclates accuracy and parameter estimation by increasing the number of lineages. """
    
    
    
    
