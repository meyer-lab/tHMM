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

abstract: Targeted therapies, such as erlotinib, operate by antagonizing dysregulated signaling nodes in tumor cells. While responses to targeted agents can be remarkable, they are rarely curative. Furthermore, cell-to-cell heterogeneity stemming from genetic mutations, variation in signaling, and epigenetic state can contribute to resistance development. Current methods of quantifying tumor-drug response are population-level measurements and lack evolutionary dynamics. We present a novel computational method, the tree-based hidden Markov model (tHMM), which utilizes cell lineages to learn characteristic patterns of single cell heterogeneity and transitions between underlying latent states for dynamic tumor cell classification. A conventional hidden Markov model was adapted to a branching binary Markov tree of dividing cells with observed emissions to classify cells based on phenotypic observations, particularly cell end-of-life fate (division or death) and lifetime. The model can, therefore, fit observations from related cells and classify subpopulations. An adapted Viterbi algorithm was built to identify the states of each cell utilizing the parameters found from the modified Baum-Welch fitting and the inheritance information from the cell linkages in the lineage tree. To probabilistically model each observation, the cell fate and lifetime follow Bernoulli ($\theta_{B}$) and exponential ($\lambda_{E}$) distributions, respectively. Synthetic lineages were constructed consisting of parent cells susceptible to drug ($\theta_{B}=0.99$, $\lambda_{E}=80$), and a later transition forming a new state distribution of resistant cells ($\theta_{B}=0.8$, $\lambda_{E}=20$). 200 such lineages of various length were constructed and analyzed, and we observed improved state assignment accuracy of the tHMM as the number of cells in a lineage increased. Parameter estimation for the Bernoulli distribution governing cell fate was precise and accurate. The exponential distribution parameter was precise but slightly biased due to systematic errors stemming from excluding cells that continued past the experiment end, as well as removing seed cells that were unable to create a lineage. The tHMM cell classification pipeline can analyze cell lineages and assign cells to phenotypically distinct subpopulations (i.e. therapy-sensitive and therapy-resistant) for a wide range of lineage lengths and population sizes. The model quantifies the probabilistic distributions governing each subpopulation and may be used in conjunction with live-cell imaging for real-time drug screening and therapy evaluation. 

link-citations: true

csl: ./manuscript/templates/nature.csl

---

# Summary points

- Current targeted therapies to stall cancer cell growth are population-based and yet to achieve single cell speficity, due to significant intratumor heterogeneity
- A tree-based hidden Markov model (tHMM) was developed to identify, and monitor over time, the pathogenic state changes of single cells in heterogenous populations
- The tHMM achieves >80% classification accuracy for lineages >50 cells. Increasing a population's number of lineages has a less noticeable effect on model accuracy, but this must be further explored for population with >20 lineages. 

# Author Summary

Cell heterogeneity, such as variability in drug response, arises as cells proliferate. _Shared_ heterogeneous traits, such as a response to drug like resistance or susceptibility within a subpopulation, are correlated across a lineage because resistant subpopulations most likely diverged from a common progenitor or a set of common progenitors that were _also_ resistant or had acquired traits leading to resistance. These acquired traits of resistance may be the result of responses to cellular microenvironments, epigenetics, and/or mutations. Using lineage tree information, we hope to capture these dynamic transitions between heterogeneous latent states of cells and arrive with a more accurate identification of cell heterogeneity in a tumor. Our computational approach employing Markov random field theory provides higher specificity through identifying intratumor resistance on an individual cell level based on lineage histories and enables real-time identification of changes in resistance throughout therapy.

# Introduction

In 2018, over 1.7 million new cases of cancer were estimated to be diagnosed with over 609,000 of those cases projected to be fatal.[@Smith; @] One of the primary treatments of cancer consists of chemotherapy, particularly targeted therapies, whereby patients are given drugs that destroy highly-prolific cells to stall cancer growth or eliminate the tumor. However, long-term therapeutic efficacy varies significantly due to the vast heterogeneity in intratumor response to therapy.[@DiMaio; @DeRoock] Specifically, each tumor consists of cancer cell subpopulations that differ in terms of cell intrinsic and extrinsic factors including genetic plasticity, epigenetic alterations, and micro-environmental stressors.[@Semenza; @Nagarsheth; @Feinberg; @Falkenberg] Current drug-screening protocols involve giving known cancer cell lines different drug doses and evaluating therapy performance based on the percent of cells eliminated and the dose required for half-maximal effect.[@Chauvin; @Zhang_L] These metrics, however, provide population averages of overall tumor response to the therapy and fail to consider the subpopulation heterogeneity. In addition, there is a need for real-time cell classification, which would identify the significant proportion of cells that undergo stochastic changes in resistance after onset of therapy.[@Gupta; @Enderling; @Inde]

Cell variability can originate from cell-intrinsic factors, such as genomic alterations (i.e. altered nucleotide excision repair and telomere maintenance), and cell-extrinsic factors such as spatial variability in vasculature and environmental stressors causing DNA promoter mutations.[@Feinberg; @Falkenberg; @Inde] Cancer cell plasticity, which is difficult to predict, further exacerbates the heterogeneity problem, as two daughter cells of the same non-resistant parent cell have been seen in extreme cases to acquire substantial therapy resistance.[@Sun] The clinical implication of cell-to-cell heterogeneity is intratumor variability in therapy resistance that leads to poor outcomes and low remission, as primary tumors that relapse after treatment possess significantly more subpopulation diversity than relapse-free tumors.[@Zhang_J] 

Recent advances in ‘omics’ technologies have allowed for detailed observation of cell-cell variability.[@DeRoock; @Gerlinger] Additionally, the development of fine mapping and protein network algorithms allow for researchers to determine the presence of causal genetic mutations and dysregulation events that drive abnormal protein function.[@Hormozdiari; @Alvarez] Although useful in detecting genetic changes, these modalities are labor and time intensive.[@Teicher] Current ‘omics’ technologies primarily serve as an end-point analysis of cells, which must be detached (i.e. prior to sequencing), preventing researchers from gaining knowledge about how cancer cell populations evolve over time. 

As an alternative, therapy resistance of subpopulations can be characterized in real-time through longitudinal, phenotypic measurements of cell fitness.[@Dallas; @Frasor; @Balic] This time-lapse cell classification of fitness markers has been demonstrated using phenotypic observations such as cell end-of-life fate, cell lifetime, and population doubling time.[@Bhadriraju; @Cerulus; @Huang_D] In fact, these cell phenotypes have been adopted in the clinical setting to understand the pathogenic nature of cells.[@Gett; @Arai; @Bourhis; @Yachida] Most efforts to phenotypically characterize cell fitness, however, are not yet resolved enough to illuminate the full complexity of cancer cells due to their reliance on population-level analysis (i.e. IC<sub>50</sub>).[@O_Connor] Recent research has made efforts to track phenotypic measurements of fitness at the single-cell level in hopes of capturing information missed by population-level analyses.[@Huang_D; @Tyson]

## Model Objective

We aim to continue in the direction of recent advances towards cell-cell phenotypic fitness measurements by constructing a novel machine learning algorithm to identify the comprehensive range of therapy resistance among intratumor cell subpopulations. Specifically, the novelty of this work lies in the tree hidden Markov model (tHMM), which can operate on any tree-structured data to perform a linkage analysis utilizing parent-daughter inheritance information. In this project, we apply the tHMM to quantitatively assess the pathogenic state for each heterogeneous cell within a lineage. Because each cell is a daughter of its parent, it can be assumed that past information from ancestors influences the phenotype of a daughter cell. A tHMM fulfills this objective of including past cellular information because it operates on a lineage tree with multiple branches instead of a group or population of cells with no information regarding inheritance, familial relationships, or individual variability. 

Other computational models for cell lineages have yet to incorporate a tHMM that provides a full spectrum of underlying cell states based on phenotypic measurements for every member in a lineage. Tree-based models utilizing linkage and inheritance information  have previously been employed in phylogeny discovery and evolutionary analyses,[@Bykova] and such models have been validated in molecular systems, such as identifying chromatin hidden states by observing histone modifications.[@Kuchen; @Biesinger] None of these models, however, have been developed for cell phenotype analysis with potential application in tumor-specific cancer therapy, nor have any of these models been provided open source for use by the scientific community. We hope to provide phenotypic measurements of cell fitness as a novel methodology for the following bedside and bench applications: (1) analyzing cells in heterogeneous cancer by providing unseen features of the inherent properties of cellular functioning and (2) providing biomedical researchers an analytical tool for understanding cellular heterogeneity within their experiments. In essence, this work will assist researchers, pathologists, oncologists, and the like in distinguishing the structure of heterogeneous cell subpopulations and identifying single-cell response to cancer therapy. 

Specifically, the tHMM expands data extrapolation for both bedside and bench use through the following functions: (1) learning to find the parameters for the tHMM given observations of lineage trees and a number of hidden states; (2) evaluating the data by finding the probability that the model generates lineage trees given some probabilistic parameters (i.e. likelihood); (3) decoding the data to find the most probable sequence of hidden states given a population; (4) predicting by discovering the next observation or sequence of observations given an initial set of lineage trees. An example protocol for sequential phenotype measurements and tHMM extrapolation are shown in Figure 1. 

![**Workflow to obtain phenotypic measurements of cell fitness for use in the tHMM.** This procedure begins using time-lapse images to trace parent-daughter linkages over an entire lineage. The lineage, possessing each cell-specific measure of lifetime and fate, is then used to model all latent states (i.e. cell subpopulations) in the sample using Bernoulli and exponential distributions that are fitted with Baum-Welch maximum likelihood estimation. After classifying each subpopulation, the pipeline assigns each cell from the lineage to its respective subpopulation (i.e. hidden state) which provides the quantitative measure for cell-specific resistance to cancer therapy. Created with BioRender.com.](./Figures/figure1.svg){#fig:workflow}

The phenotypic measurements and tHMM extrapolation are able to be conducted in real-time and do not require extensive laboratory resources characteristic of ‘omics’ techniques. Identification of cell resistance can be achieved relatively early within the growth of a cell mass or onset of a therapy, depending on the growth characteristics of the cell lineages, which are explored in later on. The tHMM pipeline in its current version is provided as an open source toolkit in Python (<https://github.com/meyer-lab/lineage-growth>), enabling other investigators to understand the identities of heterogeneous subpopulations in both basic research settings and clinical analysis. This computational tool is equipped with class modules and functions that can be easily edited to serve a user’s specific purpose but can also be used immediately. We also provide tools for lineage data preprocessing, lineage data input and output, and lineage data visualization, apart from the main methods involving tHMM analysis.




