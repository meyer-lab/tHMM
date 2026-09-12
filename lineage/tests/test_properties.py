"""Property-based tests using Hypothesis.

These tests target the core invariants of the package's data structures and
algorithms (stochastic matrices, tree structure, likelihood computations,
and the Viterbi/Baum-Welch machinery) across a wide range of randomly
generated inputs, rather than the fixed examples used in the other unit
tests.
"""

import numpy as np
import pytest
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from ..BaumWelch import calculate_stationary
from ..CellVar import CellVar
from ..LineageTree import LineageTree
from ..states.stateCommon import bern_estimator
from ..states.StateDistributionGamma import StateDistribution
from ..tHMM import tHMM

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def stochastic_matrices(draw, min_states: int = 2, max_states: int = 5):
    """Draws a square row-stochastic matrix (every row is a probability distribution)."""
    n = draw(st.integers(min_value=min_states, max_value=max_states))
    rows = []
    for _ in range(n):
        # Positive weights, normalized into a probability row. A pseudocount
        # keeps every entry bounded away from 0 and 1 so downstream sampling
        # (np.random.Generator.choice) never sees a row summing to exactly 0.
        weights = draw(
            st.lists(
                st.floats(min_value=1e-3, max_value=1.0, allow_nan=False, allow_infinity=False),
                min_size=n,
                max_size=n,
            )
        )
        weights_arr = np.array(weights)
        rows.append(weights_arr / weights_arr.sum())
    return np.vstack(rows)


@st.composite
def stochastic_vectors(draw, n: int):
    """Draws a length-n probability vector."""
    weights = draw(
        st.lists(
            st.floats(min_value=1e-3, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    weights_arr = np.array(weights)
    return weights_arr / weights_arr.sum()


@st.composite
def pi_and_T(draw, min_states: int = 2, max_states: int = 4):
    """Draws a matching (pi, T) pair with a consistent number of states."""
    T = draw(stochastic_matrices(min_states=min_states, max_states=max_states))
    pi = draw(stochastic_vectors(T.shape[0]))
    return pi, T


# ---------------------------------------------------------------------------
# CellVar.divide
# ---------------------------------------------------------------------------


@given(T=stochastic_matrices(), state=st.data(), seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_divide_preserves_tree_and_state_invariants(T, state, seed):
    """Dividing a cell must always produce two daughters with valid states,
    correct parent/gen bookkeeping, regardless of the transition matrix or
    parent state drawn."""
    parent_state = state.draw(st.integers(min_value=0, max_value=T.shape[0] - 1))
    cell = CellVar(state=parent_state, parent=None)

    left, right = cell.divide(T, rng=seed)

    for daughter in (left, right):
        assert 0 <= daughter.state < T.shape[0]
        assert daughter.parent is cell
        assert daughter.gen == cell.gen + 1
        assert daughter.isLeafBecauseTerminal()

    assert cell.left is left and cell.right is right
    assert not cell.isLeafBecauseTerminal()


@given(T=stochastic_matrices(), seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_divide_is_deterministic_given_same_rng_seed(T, seed):
    """Calling divide with the same integer seed must be reproducible,
    since downstream code (e.g. LineageTree.rand_init) relies on being able
    to seed simulations for reproducibility."""
    cell_a = CellVar(state=0, parent=None)
    cell_b = CellVar(state=0, parent=None)

    left_a, right_a = cell_a.divide(T, rng=seed)
    left_b, right_b = cell_b.divide(T, rng=seed)

    assert left_a.state == left_b.state
    assert right_a.state == right_b.state


# ---------------------------------------------------------------------------
# calculate_stationary
# ---------------------------------------------------------------------------


@given(T=stochastic_matrices())
def test_stationary_distribution_is_a_probability_vector(T):
    """For any valid transition matrix, the stationary distribution must sum to
    one and its entries, once numerical noise is trimmed, must be non-negative:
    it is a probability distribution over states."""
    pi = calculate_stationary(T)

    assert pi.shape == (T.shape[0],)
    assert np.isfinite(pi).all()
    assert np.isclose(np.sum(pi), 1.0, atol=1e-6)
    # Allow tiny negative numerical noise around zero from the eigendecomposition.
    assert np.all(pi > -1e-8)


@given(T=stochastic_matrices())
def test_stationary_distribution_is_a_fixed_point_of_T(T):
    """The defining property of a stationary distribution is w = w @ T."""
    pi = calculate_stationary(T)

    assert np.allclose(pi @ T, pi, atol=1e-5)


# ---------------------------------------------------------------------------
# bern_estimator
# ---------------------------------------------------------------------------


@given(
    bern_obs=st.lists(st.sampled_from([0.0, 1.0]), min_size=1, max_size=25),
    data=st.data(),
)
def test_bern_estimator_output_is_a_probability(bern_obs, data):
    """Regardless of the observed 0/1 outcomes and their (non-negative) weights,
    the weighted Bernoulli estimate must remain a valid probability."""
    n = len(bern_obs)
    gammas = data.draw(
        arrays(dtype=np.float64, shape=n, elements=st.floats(min_value=0.0, max_value=10.0, allow_nan=False))
    )
    bern_arr = np.array(bern_obs, dtype=float)

    p = bern_estimator(bern_arr, gammas)

    assert 0.0 <= p <= 1.0
    assert np.isfinite(p)


@given(n=st.integers(min_value=1, max_value=30))
def test_bern_estimator_all_ones_is_close_to_one(n):
    """With every observation a success and equal weights, the (pseudocount
    regularized) estimate should be pulled close to, but never past, 1."""
    bern_obs = np.ones(n, dtype=float)
    gammas = np.ones(n, dtype=float)

    p = bern_estimator(bern_obs, gammas)

    assert p < 1.0
    # The pseudocount is small (+1/+2), so for a reasonable sample size the
    # estimate should still land close to the true rate of 1.
    if n >= 10:
        assert p > 0.85


# ---------------------------------------------------------------------------
# StateDistribution (Gamma emissions) logpdf
# ---------------------------------------------------------------------------


@st.composite
def gamma_observation_rows(draw, min_size: int = 1, max_size: int = 15):
    """Draws a (bern_col, gamma_col, censor_col) observation matrix of a random
    but consistent length, as consumed by StateDistribution.logpdf."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    bern_col = draw(st.lists(st.sampled_from([0.0, 1.0, -1.0]), min_size=n, max_size=n))
    gamma_col = draw(st.lists(st.floats(min_value=-1.0, max_value=100.0, allow_nan=False), min_size=n, max_size=n))
    censor_col = draw(st.lists(st.sampled_from([0.0, 1.0]), min_size=n, max_size=n))
    return np.column_stack([bern_col, gamma_col, censor_col])


@given(
    bern_p=st.floats(min_value=0.05, max_value=0.95),
    gamma_a=st.floats(min_value=0.5, max_value=20.0),
    gamma_scale=st.floats(min_value=0.5, max_value=20.0),
    x=gamma_observation_rows(),
)
# Pinned regression example for a bug Hypothesis found: a Bernoulli sentinel of
# -1 ("unobserved") on a row whose Gamma observation is exactly 0 and uncensored,
# combined with a Gamma shape < 1 (where the density diverges to +inf at x=0),
# produces a (+inf) + (-inf) = NaN log-likelihood.
@example(bern_p=0.5, gamma_a=0.5, gamma_scale=1.0, x=np.array([[-1.0, 0.0, 1.0]]))
@pytest.mark.xfail(
    reason=(
        "Discovered by property-based testing: because lineage/__init__.py sets "
        "np.seterr(all='raise'), the NaN described above crashes with "
        "FloatingPointError instead of silently propagating. See "
        "StateDistributionGamma.logpdf, which zeroes out negative-Bernoulli rows "
        "*after* already having accumulated the diverging Gamma term into `ll`."
    ),
    strict=True,
)
def test_gamma_logpdf_never_produces_nan(bern_p, gamma_a, gamma_scale, x):
    """The log-likelihood must be a well-defined (non-NaN) number for any
    combination of valid parameters and observations, including negative or
    zero "sentinel" observations used elsewhere in the codebase to represent
    unobserved cells."""
    dist = StateDistribution(bern_p=bern_p, gamma_a=gamma_a, gamma_scale=gamma_scale)

    ll = dist.logpdf(x)

    assert ll.shape == (x.shape[0],)
    assert not np.isnan(ll).any()


@given(
    bern_p=st.floats(min_value=0.05, max_value=0.95),
    gamma_a=st.floats(min_value=0.5, max_value=20.0),
    gamma_scale=st.floats(min_value=0.5, max_value=20.0),
    n=st.integers(min_value=1, max_value=15),
    data=st.data(),
)
def test_gamma_logpdf_zero_for_negative_observations(bern_p, gamma_a, gamma_scale, n, data):
    """Negative Gamma or Bernoulli observations are a sentinel for "no
    observation" and must always contribute exactly zero log-likelihood,
    independent of the state's parameters."""
    dist = StateDistribution(bern_p=bern_p, gamma_a=gamma_a, gamma_scale=gamma_scale)

    bern_col = np.full(n, -1.0)
    gamma_col = data.draw(
        arrays(dtype=np.float64, shape=n, elements=st.floats(min_value=-100.0, max_value=-0.01, allow_nan=False))
    )
    censor_col = np.ones(n)

    x = np.column_stack([bern_col, gamma_col, censor_col])
    ll = dist.logpdf(x)

    assert np.all(ll == 0.0)


@given(
    gamma_a=st.floats(min_value=0.5, max_value=20.0),
    gamma_scale=st.floats(min_value=0.5, max_value=20.0),
)
def test_gamma_dist_to_self_is_zero(gamma_a, gamma_scale):
    """The Wasserstein-style distance between a state's distribution and
    itself must be exactly zero."""
    dist = StateDistribution(bern_p=0.5, gamma_a=gamma_a, gamma_scale=gamma_scale)
    assert dist.dist(dist) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# LineageTree.rand_init structural invariants
# ---------------------------------------------------------------------------


@given(
    pi_T=pi_and_T(min_states=2, max_states=3),
    desired_num_cells=st.integers(min_value=1, max_value=40),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(deadline=None, max_examples=30)
def test_rand_init_produces_a_valid_binary_tree(pi_T, desired_num_cells, seed):
    """For any valid (pi, T) and any requested size, the generated lineage
    must be a proper binary tree: every non-leaf has exactly two children,
    every cell's state is a valid index, and the tree has the requested
    number of cells (censoring is disabled here, so nothing gets pruned)."""
    pi, T = pi_T
    E = [StateDistribution(0.9, 7.0, 4.5) for _ in range(T.shape[0])]

    lineage = LineageTree.rand_init(pi, T, E, desired_num_cells=desired_num_cells, rng=seed)

    # rand_init grows the tree two cells at a time starting from a single
    # root, so it can only ever produce odd-sized trees; it overshoots
    # desired_num_cells by at most one cell to land on the next odd size.
    assert len(lineage) % 2 == 1
    assert desired_num_cells <= len(lineage) <= desired_num_cells + 1
    assert np.all(lineage.states >= 0)
    assert np.all(lineage.states < T.shape[0])

    n_children = np.diff(lineage.tree.indptr)
    assert np.all((n_children == 0) | (n_children == 2))

    # Every cell except the root must appear as exactly one edge's daughter.
    parents, daughters = lineage.edges
    assert len(daughters) == len(set(daughters.tolist()))
    assert 0 not in daughters or desired_num_cells == 1


@given(
    pi_T=pi_and_T(min_states=2, max_states=3),
    desired_num_cells=st.integers(min_value=1, max_value=30),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(deadline=None, max_examples=25)
def test_rand_init_is_reproducible_with_same_seed(pi_T, desired_num_cells, seed):
    """Simulating a lineage twice with the same integer seed must produce
    identical trees and states, so experiments built on rand_init are
    reproducible."""
    pi, T = pi_T
    E = [StateDistribution(0.9, 7.0, 4.5) for _ in range(T.shape[0])]

    lineage_a = LineageTree.rand_init(pi, T, E, desired_num_cells=desired_num_cells, rng=seed)
    lineage_b = LineageTree.rand_init(pi, T, E, desired_num_cells=desired_num_cells, rng=seed)

    assert np.array_equal(lineage_a.states, lineage_b.states)
    assert np.array_equal(lineage_a.tree.indptr, lineage_b.tree.indptr)
    assert np.array_equal(lineage_a.tree.indices, lineage_b.tree.indices)
    assert np.allclose(lineage_a.obs, lineage_b.obs)


# ---------------------------------------------------------------------------
# Viterbi
# ---------------------------------------------------------------------------


@given(
    pi_T=pi_and_T(min_states=2, max_states=3),
    desired_num_cells=st.integers(min_value=1, max_value=30),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(deadline=None, max_examples=25)
def test_viterbi_predicts_valid_state_sequences(pi_T, desired_num_cells, seed):
    """Whatever the (pi, T, E) parameters used to build a tHMM, the Viterbi
    decoding must return exactly one state per cell in each lineage, and
    every predicted state index must be valid."""
    pi, T = pi_T
    n_states = T.shape[0]
    E = [StateDistribution(0.9, 7.0 + i, 4.5) for i in range(n_states)]

    lineage = LineageTree.rand_init(pi, T, E, desired_num_cells=desired_num_cells, rng=seed)
    model = tHMM([lineage], num_states=n_states, fpi=pi, fT=T, fE=E)

    all_states = model.predict()

    assert len(all_states) == 1
    predicted = all_states[0]
    assert predicted.shape == (len(lineage),)
    assert np.all(predicted >= 0)
    assert np.all(predicted < n_states)


# ---------------------------------------------------------------------------
# BaumWelch log-likelihood invariants
# ---------------------------------------------------------------------------


@given(
    pi_T=pi_and_T(min_states=2, max_states=3),
    desired_num_cells=st.integers(min_value=3, max_value=25),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(deadline=None, max_examples=20)
def test_log_score_matches_viterbi_optimal_assignment(pi_T, desired_num_cells, seed):
    """The Viterbi algorithm returns the state sequence that (by
    construction) maximizes the joint log-likelihood, so scoring any other
    valid assignment of the same lineage must never produce a strictly
    higher log-score."""
    pi, T = pi_T
    n_states = T.shape[0]
    E = [StateDistribution(0.9, 7.0 + i, 4.5) for i in range(n_states)]

    lineage = LineageTree.rand_init(pi, T, E, desired_num_cells=desired_num_cells, rng=seed)
    assume(len(lineage) > 0)
    model = tHMM([lineage], num_states=n_states, fpi=pi, fT=T, fE=E)

    viterbi_states = model.predict()
    viterbi_score = model.log_score(viterbi_states)[0]

    rng = np.random.default_rng(seed + 1)
    random_states = [rng.integers(0, n_states, size=len(lineage))]
    random_score = model.log_score(random_states)[0]

    assert viterbi_score >= random_score - 1e-6
