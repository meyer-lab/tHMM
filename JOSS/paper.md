---
title: 'tHMM: A lineage tree-based hidden Markov model to quantify cellular heterogeneity and plasticity'
tags:
  - Python
  - cancer drug response
  - lineage data
  - heterogeneity
  - clustering
authors:
  - name: Adrian M. Price-Whelan^[co-first author] # note this makes a footnote saying 'co-first author'
    orcid: 0000-0003-0872-7098
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Author Without ORCID^[co-first author] # note this makes a footnote saying 'co-first author'
    affiliation: 2
  - name: Author with no affiliation^[corresponding author]
    affiliation: 3
affiliations:
 - name: Lyman Spitzer, Jr. Fellow, Princeton University
   index: 1
 - name: Institution Name
   index: 2
 - name: Independent Researcher
   index: 3
date: 13 August 2017
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Cell plasticity, which is the ability of cells to reversibly alter their phenotype, is one source of cell-to-cell heterigeneity, especially in cancer drug response and it can prevent therapies from being curative. Studies show high correlation of phenotypic traits between cells that have the same parent or grandparent, which makes it important to account for inheritence [@berge]. To comprehensively account for evolutionary processes that lead to phenomena such as heterogeneity, we need to investigate drug response in single-cell level. Single cell lineage data is a format that can provide single-cell resolution of phenotypes while preserving cell-cell relations, however, tools to explore such data is limited. Investigating cell populations given their observable traits and clustering single cells while accounting for phenotype inheritence inspires a new avenue of designing new anti-cancer regimens.

# Statement of need

`tHMM` is a Python package for exploring phenotypic heterogeneity in single-cell lineage data. It leverages the well-known principles of hidden Markov models and is adjusted to work with lineage tree data [@durand]. `tHMM` clusters cells based on their measured phenotypes and relations with other cells for improved specificity in pinpointing the structure and dynamics of variability in drug response. Integrating this model with a modular interface for defining observed phenotypes allows the model to easily be adapted to any phenotype measured in single cells.

To benchmark our model, we paired cell fate with either cell lifetimes or individual cell cycle phase lengths (G1 and S/G2) as our observed phenotypes on synthetic data and demonstrated that the model successfully classifies cells within experimentally tractable dataset sizes. As an application, we analyzed experimental measurements of cell fate and phase duration in cancer cell populations treated with chemotherapies to determine the number of distinct subpopulations [@thmm]. This `tHMM` framework allows for the flexible classification of not only single cell heterogeneity, but also any data in the form of lineage trees, such as ...?

# Mathematics

The initial probabilities of a cell being in state $k$ are represented by the vector $\pi$ that sums to 1:  
$$\pi_k = P(z_1 = k), \qquad k \in \{1, ..., K\}$$  
where $z$ shows the states and $K$ is the total number of states. The probability of state $i$ transitioning to state $j$ is represented by the $K \times K$ matrix, $T$, in which each row sums to 1:   
$$T_{i,j} = T(z_i \rightarrow z_j) = P(z_j \;| z_i), \qquad i,j \in \{1, ..., K\}$$  
The emission likelihood matrix, $EL$, is based on the cell observations. It is defined as the probability of an observation conditioned on the cell being in a specific state:  
$$EL(n,k) = P(x_n = x | z_n = k)$$
where $x_n$ shows the cell number $n$, with a total of $N$ cells in a lineage. Separate observations were assumed to be independent; for instance, cell fate is considered to be independent from the time duration of each cell phase. This facilitates calculating the likelihood of observations, such that we multiply the likelihood of all observations together for the total likelihood.

Since the hidden states are unobserved, we need an expectation-maximization (EM) algorithm, for HMMs called the Baum-Welch algorithm, to find the states and their specifications. The EM algorithm consists of two steps: (1) the expectation step (E-step) and (2) maximization step (M-step.) In the E-step, using the whole lineage tree the probability of a cell and its parent being in specific states are calculated, such that for every cell and every state we have $P(z_n = k \;| X_n) \label{E1}$ and $P(z_n = k,\; z_{n+1} = l \; | X_n) \label{E2}$. The E-step is calculated by the upward and downward recursion algorithms.

In the M-step, the distribution parameters of each state, the initial ($\pi$) probabilities, and the transition probability ($T$) matrices are estimated, given the state assignments of each cell. During fitting we switch between the E-step and the M-step and calculate the likelihood. If the likelihood stops improving below some threshold we take that to indicate that convergence has been reached. The following explains each step in detail.

## E-step
### Upward recursion

An _upward-downward_ algorithm for calculating the probabilities in hidden Markov chains (HMCs) was previously proposed by Yariv and Merhav [@yariv] which suffered from underflow. This problem was originally solved by Levinson [@levinson] for HMCs, where they adopted a heuristic based scaling, and then was improved by Devijver [@devijver] where they introduced smooth probabilities. Durand et al., [@durand] however, revised this approach for hidden Markov trees to avoid underflow when calculating $P(Z|X)$ probability matrices. To explain we need the following definitions:

- $p(n)$ is noted as the parent cell of the cell $n$, and $c(n)$ is noted as children of cell $n$.
- $\bar{X}$ is the observation of the whole tree and $\bar{X}_a$ is a subtree of $\bar{X}$ which is rooted at cell $a$.
- $\bar{Z}$ is the complete hidden state tree.
- $\bar{X}_{a/b}$ is the subtree rooted at $a$ except for the subtree rooted at cell $b$, if $\bar{X}_b$ is a subtree of $\bar{X}_a$.

For the state prediction we start by calculating the marginal state distribution (MSD) matrix. MSD is an $N \times K$ matrix that for each cell is marginalizing the transition probability over all possible current states by traversing from root to leaf cells:  
$$MSD(n,k) = P(z_{n} = k)= \sum_{i} P(z_n = k |z_{n-1} = i)\times P(z_{n-1} = i)$$

During upward recursion, the flow of upward probabilities is calculated from leaf cells to the root cells generation by generation. For leaf cells, the probabilities ($\beta$) are calculated by:  
$$\beta(n,k) = P(z_n = k\;|X_n = x_n) = \frac{EL(n,k) \times MSD(n,k)}{NF_l(n)}$$

in which $X_n$ is the leaf cell's observation, and NF (Normalizing Factor) is an $N \times 1$ matrix that is the marginal observation distribution. Since $\sum_{k} \beta_n(k) = 1$, we find the NF for leaf cells using:  
$$NF_l(n) = \sum_{k} EL(n,k) \times MSD(n,k) = P(X_n = x_n)$$

For non-leaf cells the values are given by:  
$$ \beta(n,k) = P(z_n = k\;|\bar{X}_n = \bar{x}_n) = \frac{EL(n,k) \times MSD(n,k) \times \prod_{v \in c(n)}\beta_{n,v}(k)}{NF_{nl}(n)}$$

where we calculate the non-leaf NF using:  
$$NF_{nl}(n) = \sum_{k} \Big[EL(n,k) \times MSD(n,k) \prod_{v \in c(n)} \beta_{n,v}(k)\Big]$$

and linking $\beta$ between parent-daughter cells is given by:  
$$ \beta_{p(n), n}(k) = P(\bar{X}_n = \bar{x}_n | z_{p(n)} = k) = \sum_{j} \frac{\beta_n(j) \times T_{k,j}}{MSD(n,j)}$$

By recursing from leaf to root cells, the $\beta$ and NF matrices are calculated as upward recursion. The NF matrix gives a convenient expression for the log-likelihood of the observations. For each root cell we have:

$$P(\bar{X} = \bar{x}) = \prod_{n} \frac{P(\bar{X}_n = \bar{x}_n)}{\prod_{v\in c(n)} P(\bar{X}_v = \bar{x}_v)} = \sum_{n} NF(n) \qquad n \in \{1, ..., N\}$$

The overall model log-likelihood is given by the sum over root cells:

$$log P(\bar{X} = \bar{x}) = \sum_{n} log NF(n)$$

This quantity acts as the convergence measure of the EM algorithm.

##### Downward recursion

For computing _downward recursion_, we need the following definition for each root cells:

$$ \gamma_1(k) = P(z_1 = k | \bar{X}_1 = \bar{x}_1) = \beta_1(k)$$

The other cells follow in an $N \times K$ matrix by writing the conditional probabilities as the summation over the joint probabilities of parent-daughter cells:

$$\gamma_n(k) = P(z_n = k | \bar{X}_1 = \bar{x}_1) = \frac{\beta_n(k)}{MSD(n,k)} \sum_{i}\frac{T_{i,k} \gamma_{p(n)}(i)}{\beta_{p(n),n}(i)}$$

### Viterbi algorithm

Given a sequence of observations in a hidden Markov chain, the Viterbi algorithm is commonly used to find the most likely sequence of states. Equivalently, here it returns the most likely sequence of states of the cells in a lineage tree using upward and downward recursion [@doi:10.1109/TSP.2004.832006].

Viterbi follows an upward recursion from leaf to root cells. We define $\delta$, an $N \times K$ matrix:

$$\delta (n,k) = \max\limits_{\bar{z}_{c(n)}}\{P(\bar{X}_n = \bar{x}_n, \bar{Z}_{c(n)} = \bar{z}_{c(n)} | z_n = j)\}$$

and the links between parent-daughter cells as:

$$\delta_{p(n),n}(k) = \max\limits_{\bar{z}_n} \{P(\bar{X}_n = \bar{x}_n, \bar{Z}_n = \bar{z}_n | z_{p(n)} = j)\} = \max\limits_{j}\{\delta(n,j) T_{k,j}\}$$

We initialize from the leaf cells as:

$$\delta(n,k) = P(X_n = x_n | z_n = k) = EL(n,k)$$

and for non-leaf cells use:

$$\delta(n,k) = \Big[\prod_{v \in c(n)} \delta_{n,v}(k)\Big]\times EL(n,k)$$

The probability of the optimal state tree corresponding to the observations tree, assuming root cell is noted as cell 1, is then given by:

$$Z^* = \max\limits_{k}\{\delta(1,k) \pi_k \}$$

which arises from maximization over the conditional emission likelihood (EL) probabilities by factoring out the root cells as the outer maximizing step over all possible states.

## M-step

In the M-step, we find the maximum likelihood of the hidden Markov model distribution parameters. We estimate the initial probabilities, the transition probability matrix, and the parameters of the observation distributions. The maximum likelihood estimation of the initial probabilities can be found from each state's representation in the root cells:  
$$ \pi^*_k = \gamma_1(k)$$
Similarly, the transition probability matrix is estimated by calculating the prevalence of each transition across the lineage trees:  
$$ T^*_{i,j} = \frac{\sum_{n=1}^{N-1} \xi_n(i,j)}{\sum_{n=1}^{N-1} \gamma_n(i)} $$

# Acknowledgements

This work was supported by the Jayne Koskinas Ted Giovanis Foundation for Health and Policy, NIH U01-CA215709 (A.S.M.). The authors thank Ali Farhat, Adam Weiner, and Nikan Namiri for early exploratory work.

# References