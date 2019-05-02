---
title: A lineage-based Markov tree model to quantify cellular heterogeneity

author:
- name: Spongebob Squarepants
  affilnum: a
- name: Patrick Star
  affilnum: a
- name: Squidward Tentacles
  affilnum: a
- name: Gary
  affilnum: a
- name: Mr. Krabs
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

abstract: Chemotherapy is one of the primary methods of cancer treatment due to its ability to impede the growth of highly prolific cancer cells. One of the main obstacles to successful chemotherapy treatment is that highly prolific cancer cells naturally display non-uniform responses to therapy. It therefore becomes imperative for oncologists to prescribe chemotherapy combinations which target all subpopulations of cells in a given patient’s tumor. Classifying the heterogeneous resistance of cancer cells may potentially enable the design of therapies that are tumor-specific and thus enable all malignant cells within a tumor to respond adequately and die. Current methods of intratumor cell classification possess low specificity because they use population-level measurements, such as IC<sub>50</sub>, to gauge cellular responses to cancer therapy. In addition, these methods rely on single timepoint measurements which mask cancer cell evolution dynamics. In this paper, we present a novel computational method that utilizes cell lineage trees to learn the characteristic patterns of cell heterogeneity de novo and predict variable response to drug in tumors over time. TODO METHODS. TODO RESULTS. TODO DISCUSSION.

link-citations: true

csl: ./manuscript/templates/nature.csl

---

# Summary points

- One
- Two
- Three

# Author Summary

Cell heterogeneity, such as variability in drug response, arises as cells proliferate. _Shared_ heterogeneous traits, such as a response to drug like resistance or susceptibility within a subpopulation, are correlated across a lineage because resistant subpopulations most likely diverged from a common progenitor or a set of common progenitors that were _also_ resistant or had acquired traits leading to resistance. These acquired traits of resistance may be the result of responses to cellular microenvironments, epigenetics, and/or mutations. Using lineage tree information, we hope to capture these dynamic transitions between heterogeneous latent states of cells and arrive with a more accurate identification of cell heterogeneity in a tumor. Our computational approach employing Markov random field theory provides higher specificity through identifying intratumor resistance on an individual cell level based on lineage histories and enables real-time identification of changes in resistance throughout therapy.

# Introduction

In 2018, over 1.7 million new cases of cancer were estimated to be diagnosed with over 609,000 of those cases projected to be fatal.[@SmithCancerScreening] Most primary treatments of cancer consists of chemotherapy (i.e. cytotoxic and endocrine therapies), whereby patients are given chemicals that destroy highly-prolific cells to stall cancer growth or eliminate the tumor. However, the ability of therapy to impede tumor growth varies significantly due to the vast heterogeneity in intratumor response to therapy.[@LungHeterogeneity; @ColorectalHeterogeneity] Specifically, each tumor consists of cancer cell subpopulations that differ in terms of cell intrinsic and extrinsic factors including genetic plasticity, epigenetic alterations, and micro-environmental stressors.[@BreastMicroEnv; @Chemokines; @EpigeneticMod; @HistoneInhibi] Complete cancer remission is predicated on delivering therapies that eliminate all malignant cancer subpopulations within a tumor. Current drug-screening protocols involve giving known cancer cell lines different drug doses and evaluating therapy performance based on the percent of cells killed and the dose required to have a half-maximal effect.[@chauvin2017high; @QuantitativeHistology] However, these metrics for therapy performance solely provide en bloc averages of overall tumor response to therapy that fail to consider the vast complexity of subpopulation heterogeneity. In addition, the significant proportion of cells acquiring resistance at a time point stochastically after subjection to therapy must be identified with a novel, real-time analytic method.[@GenotypicEGFR; @CGANComprehensiveMolecular; @clonalMutationalEvolution]

Intratumor heterogeneity causes distinct differences in cell phenotype among cancer cells. The cell transformations originate from underlying cell-intrinsic factors, such as genomic alterations (i.e. altered nucleotide excision repair and telomere maintenance), and cell-extrinsic factors such as spatial variability in vasculature and environmental stressors causing DNA promoter mutations.[@BreastMicroEnv; @Chemokines; @EpigeneticMod; @HistoneInhibi; @TNBpatterns] Cancer cell plasticity, which is difficult to predict, further exacerbates the heterogeneity problem, as two daughter cells of the same non-resistant parent cell have been seen in extreme cases to acquire substantial therapy resistance.[@Stochasticstatetransitions] The clinical implication of cell-to-cell heterogeneity is intratumor variability in therapy resistance that leads to poor outcomes and low remission, as primary tumors that relapse after treatment possess significantly more subpopulation diversity than relapse-free tumors.[@Cancerstemcells]


