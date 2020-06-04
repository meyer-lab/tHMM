**Overview**: A guide to the tree-hidden Markov model for analyzing heterogeneous cell lineages
===============================================================================================

Authors: Shakthi Visagan, Farnaz Mohammadi, Nikan Namiri, Adam Wiener, Ali Farhat, Alex Lim, JC Lagarde, and Aaron Meyer, PhD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We present a model to analyze populations of heterogeneous lineages, that is, lineages containing objects that belong to different states and can be clustered according to their different states. In particular, we study lineages of cells that present heterogeneous behaviour and observations. Some examples of possible input to our model are lineages of cells in a developmental sequence that start as undifferentiated stem cells and end up as clusters of differentiated cells or lineages of cells in a cancerous population rapidly differentiating over time growing resistance to different chemotherapies. We also present a way to computationally synthesize such populations.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We first introduce how to synthesize these populations by working up from the basic unit, cells. We then introduce how to synthesize lineages, which are just hierarchical lineage tree groupings of cells based on their family history. Our model ultimately analyzes populations which are groups of one or more lineages that share the same states.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

--------------

1. Synthesizing Cells
---------------------

Users will rarely ever have to create individual cells, but their
functionality is presented so the user is familiar with them. Users will
however, create lineages, and as such, they will have to become
comfortable with creating transition matrices using ``numpy``.
Transition matrices define the probabilities of how cells will divide.
Cells can only be made (new cells as a result of an existing cell
dividing) when given a transition matrix. We provide an example below.

The transition matrix defines the rate at which cells are likely to
change from one state to another. We use the defintion of a stochastic
transition rate matrix from Wikipedia, that is, the column index defines
the state in which we start, and the row defines the state at which we
end up. That is, if an element at the index of a transition matrix
:math:`T` at row :math:`i` and at column :math:`j`, is defined as
:math:`T_{i,j}`, then

.. math:: T_{i,j} = \mathbb{P}(z = j | z = i).

\ Indexing for this matrix and the states starts at :math:`0`. Usually
the number of states is represented as the capital letter :math:`K` and
indexed by the lower-case letter :math:`k`. For most of our examples, we
will deal with two states, i.e., :math:`K=2`.

.. code:: ipython3

    import numpy as np
    # The capital letter T will represent the transition matrix.
    # This transition matrix, in particular, is the two-dimensional 
    # identity matrix. Recall that the elements of the transition matrix
    # represent the probability of transitioning from one state to another.
    T = np.array([[1.0, 0.0],
                  [0.0, 1.0]],  dtype="float")
    # This identity matrix implies that there are no transitions between cells of
    # state 0 to cells of state 1. It also implies that there are no transitions 
    # between cells of state 1 to cells of state 0.
    # Another transition matrix could be the following:
    #
    # T = np.array([[0.75, 0.25],
    #               [0.15, 0.85]], dtype="float")
    #
    # Note that the rows of the transition matrix have to sum one because of the
    # Law of Total Probability and the Law of Conditional Probability. This will be
    # expanded on in a later notebook.

Cells are defined by their state, their relationships to other cells,
and collections of observations. Knowing how to create cells, however,
is not required by the user. It is beneficial to understand how the
``CellVar`` class is designed to create objects that store a ``state``,
its relationships (``left``, ``right``, ``parent``) to other cells, and
its multivariate observations (``obs``). When cells are created via the
``member`` function, their familiar relationships are automatically
assigned. The two daughters are assigned to ``left`` and ``right`` and
for the daughters, the parent is assigned to ``parent``. Other familiar
relationships can be accessed through other member variables and
functions. Note that the first generation is empty (we can see this
because the ``parent_cell`` is instantiated with ``parent=None`` and
with ``gen=1`` which means that ``gen=0`` of the lineage contains
nothing except ``None``). As such, indexing for generations of lineages
starts at ``1``. We will discuss more about the multivariate
observations that cells store later; for now, the following exercises
will focus on the transition matrix and the familiar relationships of
cells.

.. code:: ipython3

    from lineage.CellVar import CellVar as c, double
    # Question 3 will focus on the code written here.
    parent_cell = c(state=0, left=None, right=None, parent=None, gen=1)
    left_cell, right_cell = parent_cell.divide(T)

--------------

QUESTION 1:

The transition matrix above is the two-dimensional Identity matrix. What
does this imply about the transitions between cells that follow this
transition process? Are there any transitions that go from state
:math:`1` to state :math:`0`, what about from state :math:`0` to state
:math:`1`? Show that you’re correct by printing out the states of all
the cells involved. Write your answer and code below. Find the
probability of transitioning from state :math:`0` to state :math:`0`, by
accessing the element of the transition rate matrix that represents
this.




ANSWER 1:

.. code:: ipython3

    # The transition matrix being the identity matrix implies that the cells never transition between 
    # different states because the probability of doing so is 0.
    # The parent cell is state 0, so the daughters should also be state 0, because the probability of transitioning
    # from state 0 to state 0 is 1.
    
    print(parent_cell)
    print(left_cell, right_cell)
    print(f"\nThe value of the element at (0,0) of the transition rate matrix is {T[0,0]},")

--------------

QUESTION 2:

The ``gen`` argument for instantiating cells represents the generation
of the cell. Are generations in the tHMM / lineage-growth codebase
``0``-indexed or ``1``-indexed? (Do the generations of cell lineages
start at ``0`` or ``1``?) Write your answer below.




ANSWER 2:

.. code:: ipython3

    # The generations of cells start at 1. The 0-generation of a lineage contains the None.

--------------

QUESTION 3:

``parent_cell``, ``left_cell``, and ``right_cell`` define a 3 cell
lineage, with 2 generations. The first generation has one cell which was
declared and can be accessed at ``parent_cell``. Calling the member
function ``_divide`` on ``parent_cell`` created two new cells which can
be accessed at ``left_cell`` and ``right_cell``. The daughter cells of
any cell can also ALWAYS be accessed using “dot” notation, using the
member variables, ``left`` and ``right``. Note that the division process
utilizes the transition matrix. Our code provides some very basic
printing methods to print out cells. Verify that the object stored at
the ``left_cell`` and ``right_cell`` variables are the same as the
object referenced at ``parent_cell.left`` and ``parent_cell.right`` by
printing out these variables or using ``assert`` statements.




ANSWER 3:

.. code:: ipython3

    # Use the `is` keyword to compare Python objects.
    assert left_cell is parent_cell.left
    assert right_cell is parent_cell.right

--------------

QUESTION 4:

Check that ``left_cell.parent`` and ``right_cell.parent`` are equivalent
to ``parent_cell`` by printing the cells out, just as you did in
QUESTION 3.




ANSWER 4:

.. code:: ipython3

    assert left_cell.parent is right_cell.parent is parent_cell

--------------

2. Creating a synthetic lineage
-------------------------------

.. code:: ipython3

    from lineage.LineageTree import LineageTree
    from lineage.states.StateDistributionGamma import StateDistribution

Defining the :math:`\pi` initial probability vector and :math:`T` stochastic transition rate matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before, we “hard-coded” that the first cell in our lineage should be
state :math:`0`. In a Markov model, this first state (the state of the
root cell), like the states of the daughter cells, are probabilistically
expressed. These probabilities are stored in the :math:`\pi` initial
probability vector. In particular, if an element of the initial
probability vector , :math:`\pi`, at index :math:`i`, is defined as
:math:`\pi_{i}`, then

.. math:: \pi_{i}=\mathbb{P}(z_{0}=i).

\ We require for :math:`\pi` a :math:`K\times 1` list of probabilities.
These probabilities must add up to :math:`1` and they should be either
in a :math:`1`-dimensional list or a :math:`1`-dimensional numpy array.
An example is shown below.

.. code:: ipython3

    # pi: the initial probability vector
    pi = np.array([0.6, 0.4], dtype="float")
    # Recall that this means that the first cell in our lineage in generation 1 
    # has a 60% change of being state 0 and a 40% chance of being state 1.
    # The values of this vector have to add up to 1 because of the 
    # Law of Total Probability.
    
    # T: transition probability matrix
    T = np.array([[0.75, 0.25],
                  [0.25, 0.75]], dtype="float")

Defining the :math:`E` emissions matrix using state distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The emission matrix :math:`E` is a little more complicated to define
because this is where the user has complete freedom in defining what
type of observation(s) they care about. In particular, the user has to
first begin with defining what physical observation they will want to
extract from images of their cells, or test on synthetically created
lineages. For example, if one is observing kinematics or physics, they
might want to use the Gaussian distribution parameterized by a mean and
covariance to model their observations (velocity, acceleration, etc.).
If one wanted to model lifetimes of cell, one could utilize one of many
of the exponential distributions with a nonnegative support to define
time. These distributions can then be combined into a multivariate
distribution.

Ultimately, the user needs to provide three things based on the
phenotype they wish to observe, model, and predict:

1. a *probability distribution function*: a function that returns a
   **likelihood** when given a **single random observation** and
   **parameters** describing the distribution
2. a *random variable*: a function that returns **random observations**
   from the distribution when given **parameters** describing the
   distribution
3. a *estimator*: a function that returns **parameters** that describe a
   distribution when given **random observations**

These three things fundamentally define any probability distribution.

An optional boolean function can be provided to “prune” or “censor”
cells based on the observation. In our example, cells with a Bernoulli
observation of :math:`0`, which implies that the cell died, are pruned
from the tree. Another prune rule we’ve implemented is removing cells
that were born after an experimental time.

We have already built, as a starting example, a model that resembles
lineage trees of cancer cells. In our synthetic model, our emissions are
multivariate. This first emission is a Bernoulli observation, :math:`0`
implying death and :math:`1` implying division. The second emission is
continuous and are gamma distributed. Though these can be thought of
cell lifetimes or periods in a certain cell phase, we want the user to
know that these values can really mean anything and they are completely
free in choosing what the emissions and their values mean. We provide,
as mentioned above,

1. a probability distribution function that takes in as input
   multivariate samples, a Bernoulli rate parameter, and two parameters
   that define the gamma distribution, and returns a likelihood,
2. a random variable that takes in a Bernoulli parameter and two gamma
   parameters and returns multivariate samples, and
3. a estimator that returns a Bernoulli parameter and two gamma
   parameters when input multiple multivariate observations.

Finally, we also define a prune rule, as explained previously.

Ultimately, :math:`E` is defined as a :math:`K\times 1` size list of
state distribution objects. These distribution objects are rich in what
they can already do, and a user can easily add more to their
functionality. They only need to be instantiated by what parameters
define that state’s distribution.

The following code block is a standard way to define state distrbutions
and store them in an emissions list. State distributions are
instantiated via their parameters.

.. code:: ipython3

    # E: states are defined as StateDistribution objects
    
    # State 0 parameters "Resistant"
    bern_p0 = 0.99
    gamma_a0 = 7
    gamma_scale0 = 7
    
    # State 1 parameters "Susceptible"
    bern_p1 = 0.88
    gamma_a1 = 7
    gamma_scale1 = 1
    
    state_obj0 = StateDistribution(bern_p0, gamma_a0, gamma_scale0)
    state_obj1 = StateDistribution(bern_p1, gamma_a1, gamma_scale1)
    
    E = [state_obj0, state_obj1]

The final required parameters are more obvious. The first is the desired
number of cells one would like in their full unpruned lineage tree. This
can be any number. Since one of our observations is time-based, we can
also add a prune condition based on time as well. Ultimately, these
design choices are left up to the user to customize based on their state
distribution type. Without loss of generality, we provide the following
example of an ‘unpruned’ lineage tree.

.. code:: ipython3

    lineage1 = LineageTree(pi, T, E, desired_num_cells=2**5 - 1)
    # These are the minimal arguments required to instantiate lineages
    print(lineage1)
    print("\n")

In the lineage above, note that the cells now have observations. Also
note that you did not have to “hard-code” the first cell and its state.
The first observation in the observation list for each cell is a
Bernoulli observation which can either be 1 or 0. An observatioon of 1
implies that the cell lived. An observation of 0 implies that the cell
died. The second observation in the observation is the gamma observation
and represents the lifetime of the cell. Note that some cells live for
far longer than others. This is because one of the states has a
probability distribution with a gamma distribution that draws longer
times.

Analyzing our first full lineage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our project’s goal is to analyze heterogeneity. We packaged the main
capability of our codebase into one function ``Analyze``, which runs the
tree-hidden Markov Model on an appropriately formatted dataset. In the
following example, we analyze the unrpuned lineage from above.

.. code:: ipython3

    from lineage.Analyze import Analyze
    
    X = [lineage1] # population just contains one lineage
    tHMMobj, pred_states_by_lineage, LL = Analyze(X, 2) # find two states

Estimated Markov parameters (:math:`\pi`, :math:`T`, :math:`E`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let’s see how well our model estimated the parameters that created this
lineage. Recall that the model is BLIND to the true states of the cells
(unlike the code blocks above where we knew the identity of the cells
(in terms of their state)). This model primarily has to segment or
partition the tree and its cells into the number of states we think is
present in our data, and then identify the parameters that describe each
state’s distributions. We can not only check how well it estimated the
state parameters, but also the initial probability vector :math:`\pi`
and transition matrix :math:`T` vector. Note that estimating these also
get better as more lineages are added (for the :math:`\pi` vector in
particular) and in general as more cells and more lineages are added.

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
