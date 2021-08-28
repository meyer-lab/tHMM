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

`tHMM` is an open-source Python package that implements an Expectation-Maximization algorithm for a hidden Markov model to simultaneously solve for hidden states and model parameters. The purpose of this model is to identify phenotypic heterogeneity among cancer cells exposed to different concentrations of a drug, and cluster the cells based on various observations, most importantly, taking to account the cell-cell reationship in decision making. The model takes in experimental observations in the form of binary lineages of single cells in a specific format in Excel sheets (please refer to lineage/data/heiser_data/new_version to see the excel sheets for each condition). Currently, the experimental observations include the single cell fate, and cell cycle phase durations, but the model is flexible with no limitations of how many observations to use, so long as they are heritable. The model has been tested on synthetic data, and also on experimental data of AU565 breast cancer cells exposed to lapatinib and gemcitabine doses. This framework properly accounts for time and fate censorship, as the experiments run for a finite amount of time. For a thorough tutorial of the implemented method, please refer to `manuscript/05.methods.md`.

# Documentation
The `docs` folder includes a few tutorials for getting started with the package. All the functions should have a docstring explaining the purpose of the function, as well as the inputs, outputs, and the type of the variables used.

# System Requirements
## Hardware requirements
`tHMM` package requires only a standard computer with enough RAM to support the in-memory operations.

## Software requirements

### OS requirements
This package is supported for *macOS*, *Windows*, and *Linux*. The package has been tested on the following systems:
macOS: Mojave (10.14.1)
Linux: Ubuntu 16.04
Windows: 10

### Python dependencies
`tHMM` mainly depends on the Python scientific stack.

```
numpy
scipy
pandas
scikit-learn
Biopython
networkx
statsmodels
seaborn
matplotlib
```

# Installation Guide:

### Clone from GitHub
```
git clone https://github.com/meyer-lab/tHMM
```
It typically takes a few minutes to clone the repository.

# Demo and Instructions for Use

All functions for creating synthetic data, importing experimental data, and fitting are in the `lineage` folder. Each figures in the manuscript has a separate file in the `lineage/figures` folder. The synthetic observations were created under the name of state distributions in the `lineage/states` folder, and unit tests for almost all functions written in the package are in the `lineage/tests` folder.
To build figures of the manuscript, for instance figure4, you can run the following in the terminal while in the main repository folder:

```
make output/figure4.svg
```

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

for state in range(lineage.num_states):
    print("State {}:".format(state))
    print("       estimated state:", tHMMobj.estimate.E[state].params)
    print("original parameters given for state:", E[state].params)
    print("\n")
```
