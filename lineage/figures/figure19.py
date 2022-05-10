""" To plot a summary of cross validation. """
from sklearn.metrics import confusion_matrix
import itertools as it
from ..LineageTree import LineageTree, hide_observation
from ..Analyze import cv_likelihood
from ..tHMM import tHMM, fit_list
from .common import pi, T, E


def cv():
    complete_lineage = LineageTree.init_from_parameters(pi, T, E, 31)
    true_states_by_lineage = [cell.state for cell in complete_lineage.output_lineage]

    lineage = hide_observation(complete_lineage)
    tHMMobj = tHMM([lineage], 2)
    tHMMobj.fit()

    print("confusion_mat", confusion_matrix(list(it.chain(*true_states_by_lineage)), list(it.chain(*tHMMobj.predict()))))

    likelihood, all_LLs = cv_likelihood(tHMMobj, complete_lineage)
    for ix, ll in likelihood:
        print("likelihood is ", ll, " and all likelihoods are ", all_LLs[ix])