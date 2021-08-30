# tHMM

[![codecov](https://codecov.io/gh/meyer-lab/tHMM/branch/master/graph/badge.svg)](https://codecov.io/gh/meyer-lab/tHMM)
[![Documentation Status](https://readthedocs.org/projects/tHMM/badge/?version=latest)](https://lineage-growth.readthedocs.io/en/latest/?badge=latest)

[Manuscript Build](https://meyer-lab.github.io/tHMM/manuscript.html)

`tHMM` is a Python package for developing methods to quantify drug response heterogeneity, using single-cell data in the form of cell lineages.

- [Overview](#Overview)
- [Documentation](#Documentation)
- [Systems Requirements](#system-requirements)
- [Installation Guide](#Installation-Guide)
- [Demo Instructions for Use](#Demo)

# Overview

`tHMM` is an open-source Python package that implements hidden Markov models built on binary lineage trees. The purpose of this model is to identify phenotypic heterogeneity among cancer cells exposed to different concentrations of a drug, and cluster the cells based on various observations, most importantly, taking to account the cell-cell reationship in decision making. The model takes in experimental observations in the form of binary lineages of single cells in a specific Excel sheet format (please refer to lineage/data/heiser_data/new_version to see the excel sheets for each condition). Currently, the experimental observations include the single cell fate, and cell cycle phase durations, but the model is flexible with no limitations of how many observations to use, so long as they are heritable. The model has been tested on synthetic data, and also on experimental data of AU565 breast cancer cells exposed to lapatinib and gemcitabine. This framework properly accounts for time and fate censorship due to experimental limitations. For a thorough tutorial of the implemented method, please refer to `manuscript/05.methods.md`.

# Documentation
The `docs` folder includes a few tutorials for getting started with the package. All the functions should have a docstring explaining the purpose of the function, as well as the inputs, outputs, and the type of the variables used.

# System Requirements
## Hardware requirements
`tHMM` package requires only a standard computer with enough RAM to support the in-memory operations.

## Software requirements

### OS requirements
This package is supported for *macOS*, *Windows*, and *Linux*. The package has been tested on the following systems:
- macOS: Mojave (10.14.1)
- Linux: Ubuntu 20.04
- Windows: 10

### Python dependencies
`tHMM` requires `virtualenv`. All other required packages can then be installed using `make venv` to establish a virtual environment. The Python packages that will be installed are listed in `requirements.txt`.

# Installation Guide:

### Clone from GitHub
```
git clone https://github.com/meyer-lab/tHMM
```
It may take a few minutes to clone the repository.

# Demo and Instructions for Use

All functions for creating synthetic data, importing experimental data, and fitting are in the `lineage` folder. Each figures in the manuscript has a separate file in the `lineage/figures` folder. The synthetic observations were created under the name of state distributions in the `lineage/states` folder, and unit tests for almost all functions written in the package are in the `lineage/tests` folder.
 
To build figures of the manuscript, for instance figure4, you can run the following in the terminal while in the main repository folder:

```
make output/figure4.svg
```

To run the unit tests:

```
make test
```

To make the manuscript:

```
make output/manuscript.html 
```

#### Creating synthetic data and fitting the model

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

#### Importing the experimental data and fitting the model

```
import numpy as np
from lineage.LineageInputOutput import import_Heiser
from lineage.states.StateDistributionGaPhs import StateDistribution
from lineage.LineageTree import LineageTree
from lineage.Analyze import run_Analyze_over

desired_num_states = 2 # does not make a difference what number we choose for importing the data.
E = [StateDistribution() for _ in range(desired_num_states)]

# Importing only one of the replicates of control condition
control1 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00601_A5_1_V5.xlsx")]

output = run_Analyze_over([control1], 2, atonce=False)
```
To find the most likely number of states, we can calculate the AIC metrc for 1,2,3,... number of states and find out the likelihoods.
The following calculates the AIC for 2 states, as we chose in the `run_Analyze_over` above.

```
AICs = np.array([oo[0].get_AIC(oo[2], 75, atonce=True)[0] for oo in output])
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

# License
This project is covered under the MIT License.
