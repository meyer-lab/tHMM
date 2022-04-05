.. _manual-main:

Welcome to tHMM's documentation!
================================

tHMM is a Python3 library for clustering data in the form of a binary lineage tree.
This documentation includes an overview of the package, and how to use the functions and classes, 
and examples of the application of tHMM in identifying cellular phenotypic heterogeneity in cancer cells under treatment.

Features
--------

* Uses lineage tree data for clustering individuals according to their relationship to each other and measurements. 
* Outputs the number of clusters, transition probability between the clusters, and distribution corresponding to the emission for each cluster.
* Can plot the lineage trees with their associated cluster shown as a color.

Outline
-------

.. toctree:: Overview
   :name: overview
   :maxdepth: 1

   overview.rst

.. toctree:: Installation
   :name: install
   :maxdepth: 1

   installation.rst

.. toctree:: System Requirements
   :name: syst-req
   :maxdepth: 1

   system-reqs.rst

.. toctree:: Demo
   :name: demo
   :maxdepth: 1

   demo.rst

.. toctree:: Emissions
   :name: stateDistribution
   :maxdepth: 1

   stateDistributions.rst
