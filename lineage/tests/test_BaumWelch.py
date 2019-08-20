""" Unit test file. """
import unittest
import numpy as np
import scipy.stats as sp
from ..StateDistribution import StateDistribution, bernoulli_estimator, exponential_estimator, gamma_estimator, prune_rule, report_time, get_experiment_time
from ..LineageTree import LineageTree


class TestModel(unittest.TestCase):
    """ Unit tests for Baum-Welch methods. """
        
    def test_step(self):
        """ This tests that one step of Baum-Welch increases the likelihood of the fit. """
        
        # pi: the initial probability vector
        pi = np.array([0.6, 0.4], dtype="float")

        # T: transition probability matrix
        T = np.array([[0.85, 0.15],
                      [0.15, 0.85]], dtype="float")

        # State 0 parameters "Resistant"
        state0 = 0
        bern_p0 = 0.95
        gamma_a0 = 5.0
        gamma_scale0 = 1.0

        # State 1 parameters "Susciptible"
        state1 = 1
        bern_p1 = 0.85
        gamma_a1 = 10.0
        gamma_scale1 = 2.0

        state_obj0 = StateDistribution(state0, bern_p0, gamma_a0, gamma_scale0)
        state_obj1 = StateDistribution(state1, bern_p1, gamma_a1, gamma_scale1)

        E = [state_obj0, state_obj1]

        accuracies_unpruned = []
        accuracies_pruned = []
        bern_unpruned = []
        gamma_a_unpruned = []
        gamma_b_unpruned = []
        bern_pruned = []
        gamma_a_pruned = []
        gamma_b_pruned = []
        
        num = 10000

        print(num)
        # unpruned lineage
        lineage_unpruned = LineageTree(pi, T, E, num, prune_boolean=False)
        # pruned lineage
        lineage_pruned = cp.deepcopy(lineage_unpruned)
        lineage_pruned.prune_boolean = True

        X1 = [lineage_unpruned]
        X2 = [lineage_pruned]
        print("unpruned")
        
        
        tHMMobj = tHMM(X, numStates=numStates)  # build the tHMM class with X
        
        LLbefore = calculate_log_likelihood(tHMMobj, NF)
        
        fit(tHMMobj, max_iter=200)

        NF = get_leaf_Normalizing_Factors(tHMMobj)
        LL = calculate_log_likelihood(tHMMobj, NF)
        
        
        
        deltas, state_ptrs, all_states, tHMMobj, NF, LL = Analyze(X1, 2) 
        print("pruned")
        deltas2, state_ptrs2, all_states2, tHMMobj2, NF2, LL2 = Analyze(X2, 2) 
        acc1 = accuracy(X1, all_states)
        acc2 = accuracy(X2, all_states2)
        accuracies_unpruned.append(100*acc1)        
        accuracies_pruned.append(100*acc2)
        
        self.assertLess()
