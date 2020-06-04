State Distributions - How to create
-----------------------------------

Assuming the user has reviewed the Overview notebook, they should be
aware that the Emissions matrix is created using State Distribution
objects. A state can be defined as a particular condition the cell is in
(eg. Resistant vs Susceptible to drug), but also by it’s
distributions/parameters. Just as a state can be defined in whatever way
one desires, so can the distributions - typically as some type of
physical observation/phenotype.

Review the StateDistribution class below. Note that this particular
distribution uses two separate distributions corresponding to two
different measurements a particular cell can have - a bernoulli variable
to define whether the cell lives or dies, and gamma variables to define
cell lifetime. In this case, they are independent of each other, but
that may not always be the case.

.. code:: ipython3

    """ This file is completely user defined. We have provided a general starting point for the user to use as an example. """
    from math import gamma
    import numpy as np
    import scipy.stats as sp
    from numba import njit
    import scipy.special as sc
    from scipy.optimize import brentq
    
    
    from lineage.states.stateCommon import bern_pdf, bernoulli_estimator
    
    
    class StateDistribution:
        def __init__(self, bern_p, gamma_a, gamma_scale):
            """ Initialization function should take in just in the parameters for the observations that comprise the multivariate random variable emission they expect their data to have. """
            self.bern_p = bern_p
            self.gamma_a = gamma_a
            self.gamma_scale = gamma_scale
            self.params = [self.bern_p, self.gamma_a, self.gamma_scale]
    
        def rvs(self, size):  # user has to identify what the multivariate (or univariate if he or she so chooses) random variable looks like
            """ User-defined way of calculating a random variable given the parameters of the state stored in that observation's object. """
            # {
            bern_obs = sp.bernoulli.rvs(p=self.bern_p, size=size)  # bernoulli observations
            gamma_obs = sp.gamma.rvs(a=self.gamma_a, scale=self.gamma_scale, size=size)  # gamma observations
            time_censor = [1] * len(gamma_obs)  # 1 if observed
            # } is user-defined in that they have to define and maintain the order of the multivariate random variables.
            # These tuples of observations will go into the cells in the lineage tree.
            list_of_tuple_of_obs = list(map(list, zip(bern_obs, gamma_obs, time_censor)))
            return list_of_tuple_of_obs
    
        def pdf(self, tuple_of_obs):  # user has to define how to calculate the likelihood
            """ User-defined way of calculating the likelihood of the observation stored in a cell. """
            # In the case of a univariate observation, the user still has to define how the likelihood is calculated,
            # but has the ability to just return the output of a known scipy.stats.<distribution>.<{pdf,pmf}> function.
            # In the case of a multivariate observation, the user has to decide how the likelihood is calculated.
            # In our example, we assume the observation's are uncorrelated across the dimensions (across the different
            # distribution observations), so the likelihood of observing the multivariate observation is just the product of
            # the individual observation likelihoods.
    
            try:
                bern_ll = bern_pdf(tuple_of_obs[0], self.bern_p)
            except ZeroDivisionError:
                assert False, f"{tuple_of_obs[0]}, {self.bern_p}"
    
            try:
                gamma_ll = gamma_pdf(tuple_of_obs[1], self.gamma_a, self.gamma_scale)
            except ZeroDivisionError:
                assert False, f"{tuple_of_obs[1]}, {self.gamma_a}, {self.gamma_scale}"
    
            return bern_ll * gamma_ll
    
        def estimator(self, list_of_tuples_of_obs, gammas):
            """ User-defined way of estimating the parameters given a list of the tuples of observations from a group of cells. """
            # unzipping the list of tuples
            unzipped_list_of_tuples_of_obs = list(zip(*list_of_tuples_of_obs))
    
            # getting the observations as individual lists
            # {
            try:
                bern_obs = list(unzipped_list_of_tuples_of_obs[0])
                gamma_obs = list(unzipped_list_of_tuples_of_obs[1])
                gamma_censor_obs = list(unzipped_list_of_tuples_of_obs[2])
            except BaseException:
                bern_obs = []
                gamma_obs = []
                gamma_censor_obs = []
    
            bern_p_estimate = bernoulli_estimator(bern_obs, (self.bern_p,), gammas)
            gamma_a_estimate, gamma_scale_estimate = gamma_estimator(gamma_obs, gamma_censor_obs, (self.gamma_a, self.gamma_scale,), gammas)
    
            state_estimate_obj = StateDistribution(bern_p=bern_p_estimate, gamma_a=gamma_a_estimate, gamma_scale=gamma_scale_estimate)
            # } requires the user's attention.
            # Note that we return an instance of the state distribution class, but now instantiated with the parameters
            # from estimation. This is then stored in the original state distribution object which then gets updated
            # if this function runs again.
            return state_estimate_obj
    
        def tHMM_E_init(self):
            """
            Initialize a default state distribution.
            """
            return StateDistribution(0.9, 7, 3 + (1 * (np.random.uniform())))
    
        def __repr__(self):
            """
            Method to print out a state distribution object.
            """
            return "State object w/ parameters: {}, {}, {}.".format(self.bern_p, self.gamma_a, self.gamma_scale)
    
    
    # Because parameter estimation requires that estimators be written or imported,
    # the user should be able to provide
    # estimators that can solve for the parameters that describe the distributions.
    # We provide some estimators below as an example.
    # Their use in the StateDistribution class is shown in the estimator class method.
    # User must take care to define estimators that
    # can handle the case where the list of observations is empty.
    
    
    def gamma_estimator(gamma_obs, gamma_censor_obs, old_params, gammas):
        """
        This is a closed-form estimator for two parameters
        of the Gamma distribution, which is corrected for bias.
        """
        gammaCor = sum(gammas * gamma_obs) / sum(gammas)
        s = np.log(gammaCor) - sum(gammas * np.log(gamma_obs)) / sum(gammas)
        def f(k): return np.log(k) - sc.polygamma(0, k) - s
    
        if f(0.01) * f(100.0) > 0.0:
            a_hat = 10.0
        else:
            a_hat = brentq(f, 0.01, 100.0)
    
        scale_hat = gammaCor / a_hat
    
        return a_hat, scale_hat
    
    
    @njit
    def gamma_pdf(x, a, scale):
        """
        This function takes in 1 observation and gamma shape and scale parameters
        and returns the likelihood of the observation based on the gamma
        probability distribution function.
        """
        return x**(a - 1.) * np.exp(-1. * x / scale) / gamma(a) / (scale**a)


Below is an alternative version of the StateDistribution class, using a
Gaussian distribution instead of Bernoulli and Gamma. As an example,
this could represent cell size. There could, for example, be a
population of cells that are split between a luminal state and basal
state, which for the sake of this have different cell sizes (which we
assume are normally distributed). We could then define the states
through normal distributions of their size.

.. code:: ipython3

    """ This file is completely user defined. We have provided a general starting point for the user to use as an example. """
    import numpy as np
    import scipy.stats as sp
    
    
    
    class StateDistribution:
        def __init__(self, norm_loc, norm_scale):
            """ Initialization function should take in just in the parameters for the observations that comprise the multivariate random variable emission they expect their data to have. """
            self.norm_loc = norm_loc
            assert norm_scale > 0, "A non-valid scale has been given. Please provide a scale > 0"
            self.norm_scale = norm_scale
            self.params = [self.norm_loc, self.norm_scale]
    
        def rvs(self, size):  # user has to identify what the multivariate (or univariate if he or she so chooses) random variable looks like
            """ User-defined way of calculating a random variable given the parameters of the state stored in that observation's object. """
            # {
            norm_obs = sp.norm.rvs(loc=self.norm_loc, scale=self.norm_scale, size=size)  # normal observations
            #time_censor = [1] * len(gamma_obs)  # 1 if observed
            # } is user-defined in that they have to define and maintain the order of the multivariate random variables.
            # These tuples of observations will go into the cells in the lineage tree.
            list_of_tuple_of_obs = list(map(list, zip(norm_obs)))
            return list_of_tuple_of_obs
    
        def pdf(self, tuple_of_obs):  # user has to define how to calculate the likelihood
            """ User-defined way of calculating the likelihood of the observation stored in a cell. """
            # In the case of a univariate observation, the user still has to define how the likelihood is calculated,
            # but has the ability to just return the output of a known scipy.stats.<distribution>.<{pdf,pmf}> function.
            # In the case of a multivariate observation, the user has to decide how the likelihood is calculated.
            # In our example, we assume the observation's are uncorrelated across the dimensions (across the different
            # distribution observations), so the likelihood of observing the multivariate observation is just the product of
            # the individual observation likelihoods.
    
            norm_ll = sp.norm.pdf(tuple_of_obs[0], self.norm_loc, self.norm_scale)
    
            return norm_ll
    
        def estimator(self, list_of_tuples_of_obs, gammas):
            """ User-defined way of estimating the parameters given a list of the tuples of observations from a group of cells. """
            # unzipping the list of tuples
            unzipped_list_of_tuples_of_obs = list(zip(*list_of_tuples_of_obs))
    
            # getting the observations as individual lists
            # {
            try:
                norm_obs = list(unzipped_list_of_tuples_of_obs[0])
            except BaseException:
                norm_obs = []
    
            norm_loc_estimate, norm_scale_estimate = norm_estimator(norm_obs, gammas)
    
            state_estimate_obj = StateDistribution(norm_loc=norm_loc_estimate, norm_scale=norm_scale_estimate)
            # } requires the user's attention.
            # Note that we return an instance of the state distribution class, but now instantiated with the parameters
            # from estimation. This is then stored in the original state distribution object which then gets updated
            # if this function runs again.
            return state_estimate_obj
    
        def tHMM_E_init(self):
            """
            Initialize a default state distribution.
            """
            return StateDistribution(10, 1 + 10 * (np.random.uniform()))
    
        def __repr__(self):
            """
            Method to print out a state distribution object.
            """
            return "State object w/ parameters: {}, {}.".format(self.norm_loc, self.norm_scale)
    
    
    # Because parameter estimation requires that estimators be written or imported,
    # the user should be able to provide
    # estimators that can solve for the parameters that describe the distributions.
    # We provide some estimators below as an example.
    # Their use in the StateDistribution class is shown in the estimator class method.
    # User must take care to define estimators that
    # can handle the case where the list of observations is empty.
    
    
    def norm_estimator(norm_obs, gammas):
        '''This function is an estimator for the mean and standard deviation of a normal distribution, including weighting for each state'''
        mu = (sum(gammas * norm_obs) + 1e-10) / (sum(gammas)+ 1e-10)
        std = ((sum(gammas*(norm_obs-mu)**2) + 1e-10)/ (sum(gammas)+ 1e-10))**.5
        if mu == 0:
            print("mu == 0")
        if std == 0:
            print("std == 0")
        if sum(gammas) == 0:
            print("sum(gammas) == 0")
        return mu, std


The following cells compare the two StateDistributions and show how one
might make one. For the most part, the pieces are essential cut and
paste, but there is a need to understand the parts to ensure proper
creation.

First the initialization function - this should take in all defining
parameters for all distributions being used. For a normal distribution,
this would be mu - called loc in scipy functions, so I used it here -
(the population average) and the population standard deviation, or
scale. These are assigned as parameters of the class object. This is
essentially just cut and paste. One can assert that the given values
actually make sense for the distribution. For example, a normal
distribution cannot have a negative or zero standard deviation.

.. code:: ipython3

    #Bernoulli/Gamma
    def __init__(self, bern_p, gamma_a, gamma_scale):
        """ Initialization function should take in just in the parameters for the observations that comprise the multivariate random variable emission they expect their data to have. """
        self.bern_p = bern_p
        self.gamma_a = gamma_a
        self.gamma_scale = gamma_scale
        self.params = [self.bern_p, self.gamma_a, self.gamma_scale]
    
    #Normal 
    def __init__(self, norm_loc, norm_scale):
        """ Initialization function should take in just in the parameters for the observations that comprise the multivariate random variable emission they expect their data to have. """
        self.norm_loc = norm_loc
        assert norm_scale > 0, "A non-valid scale has been given. Please provide a scale > 0"
        self.norm_scale = norm_scale
        self.params = [self.norm_loc, self.norm_scale]

The next required function is one of three key functions for State
Distributions. It takes in size, which represents the number of cells in
the lineage, and assigns each one a random variable from the
characteristic distributions of that state. There is one per
distribution. The time_censor variable exists due to the time based
nature of the gamma distribution. Such a variable is unnecessary in the
normal example, but may be needed depending on the type of censoring
done.

If the variable is more complex - ie. a multivariate - the user may have
to define what that variable looks like, as stated in the function.

.. code:: ipython3

    def rvs(self, size):  # user has to identify what the multivariate (or univariate if he or she so chooses) random variable looks like
        """ User-defined way of calculating a random variable given the parameters of the state stored in that observation's object. """
        # {
        bern_obs = sp.bernoulli.rvs(p=self.bern_p, size=size)  # bernoulli observations
        gamma_obs = sp.gamma.rvs(a=self.gamma_a, scale=self.gamma_scale, size=size)  # gamma observations
        time_censor = [1] * len(gamma_obs)  # 1 if observed
        # } is user-defined in that they have to define and maintain the order of the multivariate random variables.
        # These tuples of observations will go into the cells in the lineage tree.
        list_of_tuple_of_obs = list(map(list, zip(bern_obs, gamma_obs, time_censor)))
        return list_of_tuple_of_obs
    
    def rvs(self, size):  # user has to identify what the multivariate (or univariate if he or she so chooses) random variable looks like
        """ User-defined way of calculating a random variable given the parameters of the state stored in that observation's object. """
        # {
        norm_obs = sp.norm.rvs(loc=self.norm_loc, scale=self.norm_scale, size=size)  # normal observations
        #time_censor = [1] * len(gamma_obs)  # 1 if observed
        # } is user-defined in that they have to define and maintain the order of the multivariate random variables.
        # These tuples of observations will go into the cells in the lineage tree.
        list_of_tuple_of_obs = list(map(list, zip(norm_obs)))
        return list_of_tuple_of_obs

The second key function is the probability distribution function. The
function documentation describes most of how to implement the pdf. For
univariate and independent multivariate distributions, it is fairly
simple and can just use the already implemented pdf functions in scipy.
For more complex multivariate distributions, the pdf might be more
complicated and require a custom function.

In our Gaussian example, we just return the result of the pdf, given the
StateDistribution’s parameters and the observation.

.. code:: ipython3

    def pdf(self, tuple_of_obs):  # user has to define how to calculate the likelihood
        """ User-defined way of calculating the likelihood of the observation stored in a cell. """
        # In the case of a univariate observation, the user still has to define how the likelihood is calculated,
        # but has the ability to just return the output of a known scipy.stats.<distribution>.<{pdf,pmf}> function.
        # In the case of a multivariate observation, the user has to decide how the likelihood is calculated.
        # In our example, we assume the observation's are uncorrelated across the dimensions (across the different
        # distribution observations), so the likelihood of observing the multivariate observation is just the product of
        # the individual observation likelihoods.
    
        try:
            bern_ll = bern_pdf(tuple_of_obs[0], self.bern_p)
        except ZeroDivisionError:
            assert False, f"{tuple_of_obs[0]}, {self.bern_p}"
    
        try:
            gamma_ll = gamma_pdf(tuple_of_obs[1], self.gamma_a, self.gamma_scale)
        except ZeroDivisionError:
            assert False, f"{tuple_of_obs[1]}, {self.gamma_a}, {self.gamma_scale}"
    
        return bern_ll * gamma_ll
    
    def pdf(self, tuple_of_obs):  # user has to define how to calculate the likelihood
        """ User-defined way of calculating the likelihood of the observation stored in a cell. """
        # In the case of a univariate observation, the user still has to define how the likelihood is calculated,
        # but has the ability to just return the output of a known scipy.stats.<distribution>.<{pdf,pmf}> function.
        # In the case of a multivariate observation, the user has to decide how the likelihood is calculated.
        # In our example, we assume the observation's are uncorrelated across the dimensions (across the different
        # distribution observations), so the likelihood of observing the multivariate observation is just the product of
        # the individual observation likelihoods.
        try:
            norm_ll = sp.norm.pdf(tuple_of_obs[0], self.norm_loc, self.norm_scale)
        except ZeroDivisionError:
            assert False, f"{tuple_of_obs[0]}, {self.norm_loc}, {self.norm_scale}"
        
        return norm_ll

This function is the third key StateDistribution function, used to
estimate the parameters of the distribution given only observations
(stored in cell objects in a lineage). While the previous functions are
mostly cut and paste, this one requires a bit more effort in one
specific part. Specifically, the user must define their own estimator
function. Typically this would just be the maximum likelihood estimate,
but due the the incorporation of the gammas term it is slightly more
complicated.

Namely, one must find the MLE by taking the product of the pdf over all
obervations (the likelihood), then taking the log, then the derivative
and setting equal to zero to find the optimal value. For the Bernoulli,
for example, the likelihood is the product from i=1 to n (where there
are n observations) of p^x_i \* (1-p)^1-x_i.

However, the gammas term acts as a weighting variable for each
observation as to which state it might belong to, and can be included in
the likelihood as an exponent, z_i, to which the pdf is raised. So for
the Bernoulli it becomes the product from i=1 to n of (p^x_i \*
(1-p)\ :sup:`1-x_i)`\ z_i. The MLE is then calculated as normal. Once
this modified MLE is found for each parameter, the estimator function
must simply calculate and return it given the observations and gammas.

The norm estimator function is at the bottom of the following cell. The
small values are there to correct for empty lists of observations, or
when the gammas term sums to 0.

.. code:: ipython3

    def estimator(self, list_of_tuples_of_obs, gammas):
        """ User-defined way of estimating the parameters given a list of the tuples of observations from a group of cells. """
        # unzipping the list of tuples
        unzipped_list_of_tuples_of_obs = list(zip(*list_of_tuples_of_obs))
    
        # getting the observations as individual lists
        # {
        try:
            bern_obs = list(unzipped_list_of_tuples_of_obs[0])
            gamma_obs = list(unzipped_list_of_tuples_of_obs[1])
            gamma_censor_obs = list(unzipped_list_of_tuples_of_obs[2])
        except BaseException:
            bern_obs = []
            gamma_obs = []
            gamma_censor_obs = []
    
        bern_p_estimate = bernoulli_estimator(bern_obs, (self.bern_p,), gammas)
        gamma_a_estimate, gamma_scale_estimate = gamma_estimator(gamma_obs, gamma_censor_obs, (self.gamma_a, self.gamma_scale,), gammas)
    
        state_estimate_obj = StateDistribution(bern_p=bern_p_estimate, gamma_a=gamma_a_estimate, gamma_scale=gamma_scale_estimate)
        # } requires the user's attention.
        # Note that we return an instance of the state distribution class, but now instantiated with the parameters
        # from estimation. This is then stored in the original state distribution object which then gets updated
        # if this function runs again.
        return state_estimate_obj
    
    def estimator(self, list_of_tuples_of_obs, gammas):
        """ User-defined way of estimating the parameters given a list of the tuples of observations from a group of cells. """
        # unzipping the list of tuples
        unzipped_list_of_tuples_of_obs = list(zip(*list_of_tuples_of_obs))
    
        # getting the observations as individual lists
        # {
        try:
            norm_obs = list(unzipped_list_of_tuples_of_obs[0])
        except BaseException:
            norm_obs = []
    
        norm_loc_estimate, norm_scale_estimate = norm_estimator(norm_obs, gammas)
    
        state_estimate_obj = StateDistribution(norm_loc=norm_loc_estimate, norm_scale=norm_scale_estimate)
        # } requires the user's attention.
        # Note that we return an instance of the state distribution class, but now instantiated with the parameters
        # from estimation. This is then stored in the original state distribution object which then gets updated
        # if this function runs again.
        return state_estimate_obj
    
    
    def norm_estimator(norm_obs, gammas):
        '''This function is an estimator for the mean and standard deviation of a normal distribution, including weighting for each state'''
        mu = (sum(gammas * norm_obs) + 1e-10) / (sum(gammas)+ 1e-10)
        std = ((sum(gammas*(norm_obs-mu)**2) + 1e-10)/ (sum(gammas)+ 1e-10))**.5
        if mu == 0:
            print("mu == 0")
        if std == 0:
            print("std == 0")
        if sum(gammas) == 0:
            print("sum(gammas) == 0")
        return mu, std

Lastly, one needs to make sure there is a function that creates a random
instance of the StateDistribution class, used for the tHMM. As long as
one of the parameters is random, the function should work properly in
assigning to states. If the instance was always the same, the clustering
used would not work as clusters would start off identical. Also note
that these values should make sense for the distribution. For example,
the Gaussian StateDistribution should not have the possibilty of being
created with a scale of 0. With our current class it would throw an
error, but it’s good to be safe.

The repr function merely provides instructions on how to print.
Adjusting this is merely cut and paste.

.. code:: ipython3

    def tHMM_E_init(self):
        """
        Initialize a default state distribution.
        """
        return StateDistribution(0.9, 7, 3 + (1 * (np.random.uniform())))
    
    def __repr__(self):
        """
        Method to print out a state distribution object.
        """
        return "State object w/ parameters: {}, {}, {}.".format(self.bern_p, self.gamma_a, self.gamma_scale)
    
    
    def tHMM_E_init(self):
        """
        Initialize a default state distribution.
        """
        return StateDistribution(10, 1 + 10 * (np.random.uniform()))
    
    def __repr__(self):
        """
        Method to print out a state distribution object.
        """
        return "State object w/ parameters: {}, {}.".format(self.norm_loc, self.norm_scale)

Now that we have a functioning Gaussian StateDistribution, let’s try it
with the overall model. As in the overview, we first define our pi and
transition matrices.

.. code:: ipython3

    from lineage.LineageTree import LineageTree
    pi = np.array([0.6, 0.4], dtype="float")
    
    T = np.array([[0.75, 0.25],
                  [0.25, 0.75]], dtype="float")

Per our example earlier, we have two states, corresponding to 2
different normal distributions for cell size. We then create the state
objects and Emissions matrix

.. code:: ipython3

    # E: states are defined as StateDistribution objects
    
    # State 0 parameters "Basal"
    norm_loc0 = 14
    norm_scale0 = 2
    
    # State 1 parameters "Luminal"
    norm_loc1 = 19
    norm_scale1 = 3
    
    state_obj0 = StateDistribution(norm_loc0, norm_scale0)
    state_obj1 = StateDistribution(norm_loc1, norm_scale1)
    
    E = [state_obj0, state_obj1]

Creating the lineage tree is identical. Note the observation list only
contains one random variable, instead of the 3 for the Bernoulli/Gamma
(one per distribution plus the time censor)

.. code:: ipython3

    lineage1 = LineageTree(pi, T, E, desired_num_cells=2**5 - 1)
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

The following is an analysis run on a larger set of lineages. Note that
the pi matrix is much better predicted, while the other two are also
improved. The model works, even though the State Distribution has
changed.

.. code:: ipython3

    Y = []
    for _ in range(15):
        Y.append(LineageTree(pi, T, E, desired_num_cells=2**5 - 1))
    tHMMobj, pred_states_by_lineage, LL = Analyze(Y, 2) # find two states

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

One last note - I don’t show any cell lineages that are censored/pruned
here. The reason for that is because the pruning is done by cell fate or
time currently. Neither of these apply to the Gaussian distribution, so
the censoring would not work.

