**State Distributions** - How to create
=======================================

The Emissions matrix is created using State Distribution
objects. A state can be defined as a particular condition the cell is in
(eg. Resistant vs Susceptible to a drug treatment), which is represented by it’s
distribution(s)/parameter(s). Just as a state can be defined in whatever way
one desires, so can the distributions - typically as some type of
physical observation/phenotype.

The following codes present two examples of how to create StateDistribution.
The first example shows that a state of a cell, can be determined by two phenotypes,
that are cell fate and cell lifetime. Cell fate, which is whether a cell dies or gets to divide,
is representated by a Bernoulli distribution which is defined by one parameter, shown as bern_p. 
Cell lifetime, which is the intermitotic time for each cell, is represented by a Gamma distribution
which is defined by two parameters, shown as `gamma_a` and `gamma_scale`.
In this case, they are independent of each other -- but that may not always be the case.

The second example shows that a state of a cell can be determined only using cell size.
We used a Normal distribution to model the cell size, which has two parameters;
mean and standard deviation, shown as `norm_loc` and `norm_scale`, respectively. Please note that
in this case we are assuming all cells should be accounted for, i.e., we have no censorship.

Necessary components to create StateDistribution class compatible with tHMM are listed bellow:

1. The initialization function
------------------------------

This should take in all defining
parameters for all distributions being used. For the first example, Bernoulli and Gamma, it would be
`bern_p`, `gamma_a`, and `gamma_scale`, and for the sencond example, the normal distribution,
this would be `norm_loc`, and `norm_scale`.
These are assigned as instances of the class object. One can assert that the given values
actually make sense for the distribution. For example, a normal
distribution cannot have a negative or zero standard deviation.

.. code:: ipython3
    import numpy as np
    import scipy.stats as sp

    #Bernoulli/Gamma
    def __init__(self, bern_p, gamma_a, gamma_scale):
        """ Initialization function should take in just in the parameters for the observations that comprise the multivariate random variable emission they expect their data to have. """
        self.bern_p = bern_p
        self.gamma_a = gamma_a
        self.gamma_scale = gamma_scale
        self.params = [self.bern_p, self.gamma_a, self.gamma_scale]

.. code:: ipython3
    #Normal 
    def __init__(self, norm_loc, norm_scale):
        """ Initialization function should take in just in the parameters for the observations that comprise the multivariate random variable emission they expect their data to have. """
        self.norm_loc = norm_loc
        assert norm_scale > 0, "A non-valid scale has been given. Please provide a scale > 0"
        self.norm_scale = norm_scale
        self.params = [self.norm_loc, self.norm_scale]


2. Random variable generator function
-------------------------------------

To create synthetic data, we need to generate random variables of the defined distributions as cell observations.
We do that using the rvs (random variables) function, which actually uses the built-in scipy function rvs.
It takes in size, which represents the number of cells in
the lineage, and assigns each one a random variable from the
characteristic distributions of that state. There is one per
distribution. The time_censor variable exists due to the time based
nature of the gamma distribution. Such a variable is unnecessary in the
normal example, but may be needed depending on the type of censoring
done.
While creatig our synthetic data, at first, we assume all cells are observed and there is no censored cells.
The variables `gamma_obs_censor` and `norm_obs_censor` are created and set to 1 for each cell to represent that.
We do the censoring later.
The rvs function returns the observed phenotypes as a tuple of lists.

.. code:: ipython3

    # Bernoulli/Gamma
    def rvs(self, size: int):
        """ User-defined way of calculating a random variable given the parameters of the state stored in their StateType object. """

        bern_obs = sp.bernoulli.rvs(p=self.params[0], size=size)  # bernoulli observations
        gamma_obs = sp.gamma.rvs(a=self.params[1], scale=self.params[2], size=size)  # gamma observations
        gamma_obs_censor = [1] * size  # 1 if observed

        # These tuples of observations will go into the cells in the lineage tree.
        return bern_obs, gamma_obs, gamma_obs_censor

.. code:: ipython3
    # Normal
    def rvs(self, size):  # user has to identify what the multivariate (or univariate if he or she so chooses) random variable looks like
        """ User-defined way of calculating a random variable given the parameters of the state stored in that observation's object. """

        norm_obs = sp.norm.rvs(loc=self.norm_loc, scale=self.norm_scale, size=size)  # normal observations
        norm_obs_censor = [1] * size  # 1 if observed

        # These tuples of observations will go into the cells in the lineage tree.
        return norm_obs, norm_obs_censor


3. PDF
------

The third required function is the probability distribution function (pdf).
For univariate and independent multivariate distributions, it is fairly
simple and can just use the already implemented pdf functions in scipy.
For more complex multivariate distributions, the pdf might be more
complicated and require a custom function. It is to calculate the likelihood of the observations.

In the Bernoulli/Gamma example, we assume the two phenotypes are independent
and we add their Log-lilelihood to find the total log-likelihood,
which is equivalent to multiplying the likelihoods.

In this function, we consider the censorship of the observations, 
based on the integer value we assigned to them to show whether they are censored or not.
The fully observed cells are fed to `logpdf` to calculate the likelihood,
and those cells that have missing information are fed to `logsf`.
Those cells that died are then removed in the first example that cell's fate matters.

.. code:: ipython3
    # Bernoulli/Gamma
    def pdf(self, x: np.ndarray):
        """ User-defined way of calculating the likelihood of the observation stored in a cell.
        """
        ll = np.zeros(x.shape[0])

        # Update uncensored Gamma
        ll[x[:, 2] == 1] += sp.gamma.logpdf(x[x[:, 2] == 1, 1], a=self.params[1], scale=self.params[2])

        # Update censored Gamma
        ll[x[:, 2] == 0] += sp.gamma.logsf(x[x[:, 2] == 0, 1], a=self.params[1], scale=self.params[2])

        # Remove dead cells
        ll[x[:, 0] == 0] = 0.0

        # Update for observed Bernoulli
        ll[np.isfinite(x[:, 0])] += sp.bernoulli.logpmf(x[np.isfinite(x[:, 0]), 0], self.params[0])

        return np.exp(ll)

.. code:: ipython3
    # Normal
    def pdf(self, tuple_of_obs):  # user has to define how to calculate the likelihood
        """ User-defined way of calculating the likelihood of the observation stored in a cell. """
        
        ll = np.zeros(x.shape[0])

        ll += sp.norm.pdf(tuple_of_obs[0], self.norm_loc, self.norm_scale)

        return ll


4. The estimator
----------------

The `estimator` method provides estimation of distribution parameters given the observations.
The user must define their own estimator function. 
Typically this would just be the maximum likelihood estimate,
but in our fisr example, due the the incorporation of the gammas term 
and that we have censorship, it is slightly more complicated.
One must find the MLE by taking the product of the pdf over all
obervations (the likelihood), then taking the log, then the derivative
and setting equal to zero to find the optimal value. For the Bernoulli,
for example, the likelihood is the product from i=1 to n (where there
are n observations) of p^x_i \* (1-p)^1-x_i.

For the Bernoulli it becomes the product from i=1 to n of (p^x_i \*
(1-p)\ :sup:`1-x_i)`\ z_i. The MLE is then calculated as normal. Once
this modified MLE is found for each parameter, the estimator function
must simply calculate and return it given the observations and gammas.
The Gamma estimator function takes in the observations, and uses the 
minimize function of the scipy.optimize to find the parameters and
the function is located in the lineage/states/stateCommon.py


.. code:: ipython3

    # Bernoulli/Gamma
    def estimator(self, x: list, gammas: np.array):
        """ User-defined way of estimating the parameters given a list of the tuples of observations from a group of cells. """

        # getting the observations as individual lists
        x = np.array(x)
        bern_obs = x[:, 0].astype('bool')
        γ_obs = x[:, 1]
        gamma_obs_censor = x[:, 2]

        b_mask = np.isfinite(bern_obs)
        # Both unoberved and dead cells should be removed from gamma
        g_mask = np.logical_and(np.isfinite(γ_obs), bern_obs)
        assert np.sum(g_mask) > 0, f"All the cells are eliminated from the Gamma estimator."

        # Handle an empty state
        if np.sum(gammas[b_mask]) == 0.0:
            self.params[0] = np.average(bern_obs[b_mask])
        else:
            self.params[0] = np.average(bern_obs[b_mask], weights=gammas[b_mask])

        # Don't allow Bernoulli to hit extremes
        self.params[0] = np.clip(self.params[0], 0.00001, 0.99999) # bernoulli parameter

        self.params[1], self.params[2] = gamma_estimator(γ_obs[g_mask], gamma_obs_censor[g_mask], gammas[g_mask], self.params[1:3]) # gamma shape and scale

.. code:: ipython3
    # Normal
    def estimator(self, x: list, gammas: np.array):
        """ User-defined way of estimating the parameters given a list of the tuples of observations from a group of cells. """

        # getting the observations as individual lists
        x = np.array(x)
        norm_obs = x[:, 0]
        norm_obs_censor = x[:, 2]

        # mask for shape
        s_mask = np.isfinite(norm_obs)
        assert np.sum(s_mask) > 0, f"All the cells are eliminated from the Gamma estimator."
        self.params[0] = np.average(norm_obs[s_mask], weights=gammas[s_mask]) # mean
        self.params[1] = ((np.sum(gammas[s_mask]*(norm_obs-self.params[0])**2) + 1e-10)/ (np.sum(gammas[s_mask])+ 1e-10))**.5 # std


Example
-------

Now that we have a functioning Gaussian StateDistribution, let’s try it
with the overall model. As in the overview, we first define our initial probability vector and
the state transition probability matrices.

.. code:: ipython3

    from lineage.LineageTree import LineageTree

    pi = np.array([0.6, 0.4], dtype="float")
    
    T = np.array([[0.75, 0.25],
                  [0.25, 0.75]], dtype="float")

Creating the Emissions for two states:

.. code:: ipython3

    # E: states are defined as StateDistribution objects
    
    # Normal distribution state 0 parameters "Basal"
    norm_loc0 = 14
    norm_scale0 = 2
    
    # Normal distribution state 1 parameters "Luminal"
    norm_loc1 = 19
    norm_scale1 = 3
    
    state_obj0 = StateDistribution(norm_loc0, norm_scale0)
    state_obj1 = StateDistribution(norm_loc1, norm_scale1)
    
    E = [state_obj0, state_obj1]

Creating the lineage tree:

.. code:: ipython3

    lineage1 = LineageTree.init_from_parameters(pi, T, E, desired_num_cells=2**5 - 1)
    # These are the minimal arguments required to instantiate lineages
    print(lineage1)
    print("\n")

Below is the analysis for a single lineage. Note that the state objects
are merely switched. However, the model fairly accurately predicts the
transition matrix and state parameters.

.. code:: ipython3

    from lineage.Analyze import Analyze
    X = [lineage1] # population just contains one lineage
    tHMMobj, pred_states_by_lineage, LL = Analyze(X, 2) # find two states

.. code:: ipython3

    print(tHMMobj.estimate.pi)

.. code:: ipython3

    print(tHMMobj.estimate.T)

.. code:: ipython3

    for state in range(lineage1.num_states):
        print("State {}:".format(state))
        print("                    estimated state:", tHMMobj.estimate.E[state])
        print("original parameters given for state:", E[state])
        print("\n")
