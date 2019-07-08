---
title: A lineage-based Markov tree model to quantify cellular heterogeneity
author:
- name: Shakthi Visagan
  affilnum: a
- name: Nikan K. Namiri
  affilnum: a
- name: Ali Farhat
  affilnum: a
- name: Adam Weiner
  affilnum: a
- name: Farnaz Mohammadi
  affilnum: a
- name: Aaron S. Meyer
  affilnum: a,b
keywords: [cancer, heterogeneity, lineage, hidden Markov Model]
affiliation:
- name: Department of Bioengineering, Jonsson Comprehensive Cancer Center, Eli and Edythe Broad Center of Regenerative Medicine and Stem Cell Research; University of California, Los Angeles
  key: a
- name: Contact info
  key: b
bibliography: ./manuscript/references.bib
abstract: Cell plasticity, or the ability of cells within a population to reversibly alter epigenetic state, is an important feature of tissue homeostasis during processes such as wound healing and is dysregulated in cancer. Plasticity cooperates with other sources of cell-cell heterogeneity, including genetic mutations and variation in signaling, during resistance development. Ultimately these mechanisms prevent most cancer therapies from being curative. Methods of quantifying tumor-drug response predominantly operate on population-level measurements and therefore lack evolutionary dynamics, which are particularly critical for highly dynamic processes such as plasticity. Here, we apply tree-based adaptation of a hidden Markov model (tHMM), utilizing single cell lineages, to learn characteristic patterns of single cell heterogeneity and state transitions. This model enables single-cell classification based on the phenotype of individual cells and their relatives for improved specificity when pinpointing the molecular drivers of variability in drug response. Integrating this model with a probabilistic language for defining observed phenotypes enabled flexible phenotype specification. Using only cell growth and death as our observed phenotype on synthetic data demonstrated that the model successfully classifies cells within experimentally-tractable dataset sizes. A model accounting for cell cycle phase successfully identified resistant cells within a heterogeneous breast cancer population, which matched with molecular markers of resistance in those same cells. In total, this tHMM framework allows for flexible classification of single-cell heterogeneity across heritable phenotypes.
link-citations: true
csl: ./common/templates/nature.csl
---

# Summary points

- A tree-based hidden Markov model (tHMM) captures cell-cell variability and dynamic population changes.
- Using a probabilistic language to define observed phenotypes allows the model to work with a wide variety of single-cell measurements.
- The model successfully classifies cells within experimentally-tractable dataset sizes.
- Classifying cells based on their phenotypic heterogeneity can uncover resistance mechanisms masked at the population level.

# Author Summary

Cell heterogeneity, such as variability in drug response, arises as cells proliferate. _Shared_ heterogeneous traits, such as a response to drug like resistance or susceptibility within a subpopulation, are correlated across a lineage because resistant subpopulations most likely diverged from a common progenitor or a set of common progenitors that were _also_ resistant or had acquired traits leading to resistance. These acquired traits of resistance may be the result of responses to cellular microenvironments, epigenetics, and/or mutations. Using lineage tree information, we hope to capture these dynamic transitions between heterogeneous latent states of cells and arrive with a more accurate identification of cell heterogeneity in a tumor. Our computational approach employing Markov random field theory provides higher specificity through identifying intratumor resistance on an individual cell level based on lineage histories and enables real-time identification of changes in resistance throughout therapy.

# Introduction

## Tumor Heterogeneity

In 2018, over 1.7 million new cases of cancer were estimated to be diagnosed with over 609,000 of those cases projected to be fatal.[@Smith] One of the primary treatments of cancer consists of chemotherapy, particularly targeted therapies, whereby patients are given drugs that destroy highly-prolific cells to stall cancer growth or eliminate the tumor. Long-term therapeutic efficacy, however, varies significantly due to the vast heterogeneity in intratumor response to therapy [@DiMaio; @DeRoock]. Cell variability can originate from cell-intrinsic factors, such as genomic alterations (i.e. altered nucleotide excision repair and telomere maintenance), and cell-extrinsic factors such as spatial variability in vasculature and environmental stressors causing DNA promoter mutations.[@Feinberg; @Falkenberg; @Inde]. Current drug-screening protocols involve subjecting studied cancer cell lines to variable drug doses and evaluating therapy performance based (1) percent of cells eliminated and (2) dose required for half-maximal effect [@Chauvin; @Zhang_L]. These metrics, however, provide population average tumor response to therapy that fails to consider subpopulation heterogeneity. 

Moreover, there is yet a modality that produces these tumor response metrics in real-time, which would enable identification of cells that undergo stochastic changes in resistance throughout therapy [@Gupta; @Enderling; @Inde]. Specifically, each tumor consists of subpopulations that constantly evolve due to cell intrinsic and extrinsic factors including genetic plasticity, epigenetic alterations, and micro-environmental stressors [@Semenza; @Nagarsheth; @Feinberg; @Falkenberg]. In the extreme case, two daughter cells of the same non-resistant parent cell have been seen to acquire substantial non-overlapping therapy resistance.[@Sun] As a result, these newly-resistant progeny can potentially repopulate the tumor population. [@risom2018differentiation]. Failing to identify such stochastic heterogeneity in real-time leads to low remission, as primary tumors that relapse after treatment possess significantly more subpopulation diversity than those that are relapse-free.[@Zhang_J] 

## Current Methods

Recent advances in ‘omics’ technologies have allowed for detailed observation of cell-cell variability.[@DeRoock; @Gerlinger] Additionally, the development of fine mapping and protein network algorithms have determined the presence of causal genetic mutations and dysregulation events that drive abnormal protein function.[@Hormozdiari; @Alvarez] Although useful in detecting genetic changes, these modalities are labor and time-intensive.[@Teicher] As a result, current ‘omics’ technologies serve primarily as end-point analysis of cells, which must be detached from their native environemnt (i.e. prior to sequencing), barring any further understanding about their evolution. Furthermore, genetic _association_ studies (i.e. Cancer Cell Line Encyclopedia) use population-level samples to find common risk factors with smaller effect sizes. The findings are valuable but yet lack the ability to find rare and meaningful transitions. [@barretina2012cancer], in particular the low probability stochastic changes in individual cell state that have large effects on overall tumor resistance.

### Phenotypic measurements of Cell Fitness

As an alternative, therapy response of subpopulations can be characterized in near real-time through longitudinal, phenotypic measurements of cell fitness.[@Dallas; @Frasor; @Balic] Fitness markers such as cell end-of-life fate, lifetime, and population doubling time have been obtained using time-lapse imaging, [@Bhadriraju; @Cerulus; @Huang_D] and are adopeted in the clincal setting to measure cell pathologies.[@Gett; @Arai; @Bourhis; @Yachida] Recent research has made efforts to track phenotypic measurements of fitness at the single-cell level [@Huang_D; @Tyson]; however, most efforts are not yet resolved enough to illuminate the full complexity of cancer cells in large part due to reliance on population-level analysis (i.e. IC<sub>50</sub>) [@O_Connor].

Evidently,phenotypic measuremnts cannot illuminate the scope of resistance inheritance by themselves, but rather require an additiaonl means of tracking inheritance patterns through multiple generations of cells. The latter concept has been well-implemented in genetic studies of _linkage_ analyses. These linkage studies use pairs of affected relatives (i.e. siblings, parents) to identify genes that are shared more frequently among those who exhibit a known phenotype [@concannon2009genetics]. As a result, linkage analyses have found success in identifying rare risk factors that possess large effect sizes. Of note, linkage analyses are most powerful when multiple routes to molecular change can each give rise to a common phenotypic change. For example, if mutating seven different genes can all give rise separately to a particular disease, analyzing each route conveys the true rise to pathogenesis, rather than a less meaningful average.

## Tree Hidden Markov Model

In the cellular case, linkage analyses rely on lineage trees to identify groups of related cells within subpopulations and in general, hidden Markov models (HMMs) let one infer cell state from indirect measurements. By expanding the HMM framework to a lineage tree, we can form a tree Hidden Markov model (tHMM) to enable single-cell linkage analysis of cellular heterogeniety. We continue in the direction of recent advances towards cell-cell phenotypic fitness measurements by constructing a machine learning algorithm to identify linked inheritance of therapy resistance among individual cells within tumor subpopulations. 

The novelty of this work lies in the tHMM, which can operate on any tree-structured data to perform a linkage analysis utilizing parent-daughter inheritance information. Here, we apply the tHMM to quantitatively assess the pathogenic state for each heterogeneous cell within a lineage. Because each cell is a daughter of its parent, it can be assumed that past information from ancestors influences the phenotype of a daughter cell. A tHMM fulfills this objective of including past cellular information because it operates on a lineage tree with multiple branches instead of a group or population of cells with no information regarding inheritance, familial relationships, or individual variability. 

### Previous Implementations

Other computational models for cell lineages have yet to incorporate a tHMM that provides a full spectrum of underlying cell states based on phenotypic measurements for every member in a lineage. Tree-based models utilizing linkage and inheritance information have previously been employed in phylogeny discovery and evolutionary analyses,[@Bykova] and such models have been validated in molecular systems, such as identifying chromatin hidden states by observing histone modifications.[@Kuchen; @Biesinger] Previously, stem cell lineages have been modelled by hidden Markov trees for the purpose of reconstructing the lineage tree based on the observations of the level of expression of a particular cell surface antigen. [@vbemHMT] More recently, others have shown success in inferring the latent states of bacterial cells and the associated state-switching dynamics from lineage tree data. [@nakashima2018deciphering]

In general, hidden Markov tree algorithms have been used for a multitude of purposes, from document image classification to understanding plant-tree architecture. [@diligenti2003hidden; @durand2005analysis] Many of these tHMMs use inference (expectation-maximization and belief propagation) algorithms for tree restoration based originally from the algorithms created for hidden Markov chains, with minor modifications to how state information is propagated through the tree; namely, Crouse et al developed an upward-downward recursion expectation-maximization algorithm different from the original forward-backward algorithm usually associated with HMMs for finding wavelet coefficients in statistical signal processing. [@crouse1998wavelet; @durand2004computational]

None of these models, however, have been developed for cell phenotype analysis with potential application in tumor-specific cancer therapy, nor have any of these models been provided open source for use by the scientific community. We hope to provide phenotypic measurements of cell fitness as a novel methodology for the following bedside and bench applications: (1) analyzing cells in heterogeneous cancer by providing unseen features of the inherent properties of cellular functioning and (2) providing biomedical researchers an analytical tool for understanding cellular heterogeneity within their experiments. In essence, this work will assist researchers, pathologists, oncologists, and the like in distinguishing the structure of heterogeneous cell subpopulations and identifying single-cell response to cancer therapy. 

### Model Objective

Specifically, the tHMM expands phenotypic cell classification through the following functions: (1) learning to find the parameters for the tHMM given observations of lineage trees and a number of hidden states; (2) evaluating the data by finding the probability that the model generates lineage trees given some probabilistic parameters (i.e. likelihood); (3) decoding the data to find the most probable sequence of hidden states given a population; (4) predicting by discovering the next observation or sequence of observations given an initial set of lineage trees. An example protocol for sequential phenotype measurements and tHMM extrapolation are shown in Figure 1. 

![**Workflow to obtain phenotypic measurements of cell fitness for use in the tHMM.** This procedure begins using time-lapse images to trace parent-daughter linkages over an entire lineage. The lineage, possessing each cell-specific measure of lifetime and fate, is then used to model all latent states (i.e. cell subpopulations) in the sample using Bernoulli and exponential distributions that are fitted with Baum-Welch maximum likelihood estimation. After classifying each subpopulation, the pipeline assigns each cell from the lineage to its respective subpopulation (i.e. hidden state) which provides the quantitative measure for cell-specific resistance to cancer therapy. Created with BioRender.com.](./figures/figure1.svg){#fig:workflow}

The phenotypic measurements and tHMM extrapolation are able to be conducted in real-time and do not require extensive laboratory resources characteristic of ‘omics’ techniques. Identification of cell resistance can be achieved relatively early within the growth of a cell mass or onset of a therapy, depending on the growth characteristics of the cell lineages, which are explored below. The tHMM pipeline in its current version is provided as an open source toolkit in Python (<https://github.com/meyer-lab/lineage-growth>), enabling other investigators to understand the identities of heterogeneous subpopulations in both basic research settings and clinical analysis. This computational tool is equipped with class modules and functions that can be easily edited to serve a user’s specific purpose but can also be used immediately. We also provide tools for lineage data preprocessing, lineage data input and output, and lineage data visualization, apart from the main methods involving tHMM analysis.




