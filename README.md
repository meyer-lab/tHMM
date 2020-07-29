# Developing methods to quantify drug response heterogeneity

[![Build Status](https://transduc.seas.ucla.edu/buildStatus/icon?job=meyer-lab/tHMM/master)](https://transduc.seas.ucla.edu/job/meyer-lab/job/tHMM/job/master/)
[![codecov](https://codecov.io/gh/meyer-lab/tHMM/branch/master/graph/badge.svg)](https://codecov.io/gh/meyer-lab/tHMM)
[![Documentation Status](https://readthedocs.org/projects/tHMM/badge/?version=latest)](https://lineage-growth.readthedocs.io/en/latest/?badge=latest)

Quantifying the degree of cell response to a drug is foundational to new drug development and clinical application. Even in the earliest stages of drug development, screening a drug’s effect on cells is often used to select compounds for further investigation. After drug approval, a compound’s response across cell lines is used to identify molecular subtypes of cancer that predict therapeutic response, study therapeutic resistance, and identify more effective drug combinations. In nearly every case, and especially in cancer, drug response can be heterogeneous due to variation in the state of individual cells. Despite this, methods currently do not exist to quantify heterogeneity in drug response directly.

While existing methods of quantifying drug response (such as fitting to the Hill equation) can be effective in capturing an overall average effect, doing so masks important potential differences in the effect of a compound. For example, in the presence of heterogeneity with a compound that inhibits growth, at least three situations can exist: (1) The compound could have an equal effect across cell subpopulations. (2) The compound could exclusively influence growth within one subpopulation. (3) The compound could shift the relative abundance of subpopulations. Each of these situations has consequences for the potential biological mechanisms of resistance as well as how one might consider drug combinations. For example, in the second situation, one should focus efforts on drugs with the maximal effect in the unaffected subpopulation.

The focus of this Capstone Design project is to develop the experimental and computational analysis framework to quantify the degree of heterogeneity in drug response. To do so, this project borrows from the evolutionary theory of applying hidden Markov models to lineage trees. Approximately two-thirds of this project is computational, with the experimental portion focused on collecting data for and verifying outcomes of the computational analysis. The experimental portion will involve automated, high-throughput live-cell imaging and image analysis to collect drug response information, along with immunofluorescence to verify predictions of cell heterogeneity. The goals of this project are as follows:

1.	Build a model of cell growth assuming the absence of cell-cell heterogeneity
2.	Extend this model to one involving cell-cell heterogeneity with the identity of each cell captured as a hidden state
3.	Apply this analysis framework to study a breast cancer cell line’s response to chemotherapy

Accomplishing these goals will create a basis on which one can study heterogeneity in drug response. Correspondingly, it will apply these methods for a disease in which heterogeneity is known to be a barrier to curative therapies.
