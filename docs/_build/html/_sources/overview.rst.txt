.. _Overview:

.. highlight:: shell

========
Overview
========

We present `tHMM`, a Python3 library for exploring heterogeneity in the binary tree
lineage format. `tHMM` leverages the well-known principles of hidden Markov models and
is adjusted to work with lineage tree data. `tHMM` performs clustering of the
individuals, based on their specific measurements and relations with other individuals
for improved specificity in pinpointing the structure and dynamics of heterogeneity
focused on the measured phenotypes. The lineage tree of individuals is implemented as
a Python class where each individual is an object, and its properties are defined as
class methods and instances. The structure of the lineage tree in our model is
composed of two layers, one is the tree of states, which is a class that stores the state
of individuals in the lineage tree structure, and on top of it there is the tree of
observations which stores the measured phenotype of each individual in the lineage
tree. `tHMM` uses the tree of observations and applies an expectation maximization (E-
M) approach including Baum Welch and Viterbi that have been modified to work with
the tree-based structure of the data to update the states and their distribution at each E-
M step. Each measurement is defined as a distribution that could be continuous or
discrete. Notably, the states (clusters) are different in their distribution parameters.
`tHMM` is flexible such that it can use any type of observations as long as there is a
distribution that can be fit to the measurements. Since this is a clustering approach, we
used Bayesian Information Criterion (BIC) to determine the number of clusters with the
highest likelihood. We incorporated Python packages such as Networkx and Bio to
visualize the lineage trees with their custom states. We benchmarked our model using
synthetic and experimental data of cellular lineages treated with chemotherapy and
growth factors and explored the phenotypic heterogeneity within the cell populations
according to their observed fates and measured cell cycle phase durations. Overall,
`tHMM` can be employed to explore heterogeneity within a lineage-like data where
individuals partially inherit their traits.