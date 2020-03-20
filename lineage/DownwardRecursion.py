"""File holds the code for the downward recursion."""

import numpy as np
from .UpwardRecursion import beta_parent_child_func


def get_root_gammas(tHMMobj, betas):
    """need the first gamma terms in the baum welch, which are just the beta values of the root nodes."""
    gammas = []

    for num, lineageObj in enumerate(tHMMobj.X):  # for each lineage in our Population
        gamma_array = np.zeros((len(lineageObj.output_lineage), tHMMobj.num_states))
        gamma_array[0, :] = betas[num][0, :]
        gammas.append(gamma_array)

    for _, gamma in enumerate(gammas):  # for each lineage in our Population
        assert np.isclose(np.sum(gamma[0]), 1.0)

    return gammas


def get_nonroot_gammas(tHMMobj, gammas, betas):
    """get the gammas for all other nodes using recursion from the root nodes"""
    T = tHMMobj.estimate.T

    for num, lineageObj in enumerate(tHMMobj.X):  # for each lineage in our Population
        lineage = lineageObj.output_lineage

        with np.errstate(divide="ignore", invalid="ignore"):
            coeffs = betas[num] / tHMMobj.MSD[num]

        for level in lineageObj.output_list_of_gens[1:]:
            for cell in level:
                parent_idx = lineage.index(cell)

                for daughter in cell.get_daughters():
                    child_idx = lineage.index(daughter)

                    beta_parent = beta_parent_child_func(beta_array=betas[num], T=T, MSD_array=tHMMobj.MSD[num], node_child_n_idx=child_idx)

                    sum_holder = np.matmul(gammas[num][parent_idx, :] / beta_parent, T)
                    gammas[num][child_idx, :] = coeffs[child_idx, :] * sum_holder

        assert np.all(gammas[num][0, :] == betas[num][0, :])

    for _, gg in enumerate(gammas):
        assert np.allclose(np.sum(gg, axis=1), 1.0)
