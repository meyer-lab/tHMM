.. _Demo:

.. highlight:: shell

=====================
Demo and Instructions
=====================

All functions for creating synthetic data, importing experimental data, and fitting are in the `lineage` folder. Benchmarking figure function that can be accessed through `lineage/figures` folder. The synthetic cellular observations were created under the name of state distributions in the `lineage/states` folder, and unit tests for almost all functions written in the package are in the `lineage/tests` folder.
 
To build the figures, for instance figure4, you can run the following in the terminal while in the main repository folder:

```
make output/figure4.svg
```

To run the unit tests:

```
make test
```


Creating synthetic data and fitting the model
---------------------------------------------

The following shows how to create a 2-state synthetic lineage of cells with cell fate and cell lifetime observations, fit them to the model, and output the corresponding transition matrix, initial probability matrix, and the estimated parameters for the distribution of each state.

```
import numpy as np
from lineage.states.StateDistributionGamma import StateDistribution
from lineage.LineageTree import LineageTree

# pi: the initial probability vector
pi = np.array([0.6, 0.4], dtype="float")
# This means that the first cell in our lineage in generation 1
# has a 60% change of being state 0 and a 40% chance of being state 1.
# The values of this vector have to add up to 1 because of the
# Law of Total Probability.

# T: transition probability matrix
T = np.array([[0.75, 0.25],
              [0.25, 0.75]], dtype="float")

# State 0 parameters "Resistant"
bern_p0 = 0.99 # a cell fate parameter (Bernoulli distribution)
gamma_a0 = 7 # the shape parameter of the gamma distribution, corresponding to the cell cycle duration
gamma_scale0 = 7 # the scale parameter of the gamma distribution, corresponding to the cell cycle duration

# State 1 parameters "Susceptible"
bern_p1 = 0.88 # a cell fate parameter (Bernoulli distribution)
gamma_a1 = 7 # the shape parameter of the gamma distribution, corresponding to the cell cycle duration
gamma_scale1 = 1 # the scale parameter of the gamma distribution, corresponding to the cell cycle duration

state_obj0 = StateDistribution(bern_p0, gamma_a0, gamma_scale0)
state_obj1 = StateDistribution(bern_p1, gamma_a1, gamma_scale1)

E = [state_obj0, state_obj1]

# creating the synthetic lineage of 15 cells, given the state distributions, transition probability, and the initial probability vector.
lineage = LineageTree.init_from_parameters(pi, T, E, desired_num_cells=2**9 - 1)
```

Now that we have created the lineages as Python objects, we use the following function to fit this data into the model.

```
from lineage.Analyze import Analyze

X = [lineage] # population just contains one lineage
tHMMobj, pred_states_by_lineage, LL = Analyze(X, 2) # find two states

# Estimating the initial probability vector
print(tHMMobj.estimate.pi)

# Estimating the transition probability matrix
print(tHMMobj.estimate.T)

# The total log likelihood
print(LL)

for state in range(lineage.num_states):
    print("State {}:".format(state))
    print("       estimated state:", tHMMobj.estimate.E[state].params)
    print("original parameters given for state:", E[state].params)
    print("\n")
```

Importing the experimental data and fitting the model
-----------------------------------------------------

```
import numpy as np
from lineage.LineageInputOutput import import_exp_data
from lineage.states.StateDistributionGaPhs import StateDistribution
from lineage.LineageTree import LineageTree
from lineage.Analyze import run_Analyze_over

desired_num_states = 2 # does not make a difference what number we choose for importing the data.
E = [StateDistribution() for _ in range(desired_num_states)]

# Importing only one of the replicates of control condition
control1 = [LineageTree(list_of_cells, E) for list_of_cells in import_exp_data(path=r"lineage/data/LineageData/AU00601_A5_1_V5.xlsx")]

output = run_Analyze_over([control1], 2, atonce=False)
```
To find the most likely number of states, we can calculate the BIC metrc for 1,2,3,... number of states and find out the likelihoods.
The following calculates the BIC for 2 states, as we chose in the `run_Analyze_over` above.

```
BICs = np.array([oo[0].get_BIC(oo[2], 75, atonce=True)[0] for oo in output])
```

The output of fitting could be the transition matrix:
```
np.array([oo[0].estimate.T for oo in output])
```

initial probability matrix:
```
np.array([oo[0].estimate.pi for oo in output])
```

the assigned cell states lineage by lineage:
```
np.array([oo[1] for oo in output])
```

the distribution parameters for each state:
```
for state in range(2):
    print("State {}:".format(state))
    print("       estimated state:", output[0][0].estimate.E[state].params)
    print("\n")
```

Depending on the number of cells and lineages being used for fitting, the run time for `Analyze` and other similar functions that run the fitting, could takes minutes to hours.
