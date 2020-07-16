## Results

### A tree hidden Markov model can disentangle single-cell states in a population, given only heritable observations.

TODO at last.

### tHMM can be trained on several observable phenotypes as emissions.
Any observable heritable phenotype with a definable distribution could be used as an emission. The following (Figure 2) shows a schematic of a lineage tree in which we can observe phenotypes such as, cell fate, inter-mitotic times (lifetimes), G1 cell cycle phase duration, G2 cell cycle phase duration, and cell shape. For each of these features we consider a probabilistic distribution, shown in parts b-e.

![**This figure shows the flexibility of the model and that we can use any tracktable phenotype.](./output/figure2.svg){#fig:2}

### tHMM performs accurately on experimentally-reasonable large lineages.
Here we show the performance of the model by increasing the cell numbers. The figure shows the state assignment accuracy and parameter estimations for censored lineages.

![**A figure to show how many cells we would need to obtain a reasonable performance.](./output/figure3.svg){#fig:3}

### tHMM performs successfully on a population with reasonably different states.
Employing Wasserstein distance, we show the more different the states, the higher the accuracy of state assignments.

![**Wasserstein distance shows in a 2-state population, the accuracy increases if the states are distant from each other.](./output/figure4.svg){#fig:4}

### tHMM can pick up even rare phenotypes within the lineage.

This parts explains how the model performs in the case of an under-represented population of cells.

![**State assignment accuracy when we have different proportions of each state, showing the model performs well even when there is an under- (or over-) represented population.](./output/figure5.svg){#fig:5}

### AIC metric indicates the model's capability to detemine the closest number of states in a population.

In this part we explain the AIC figures.

![**AIC of the model varying the number of states predicted, showing that AIC can be used as a metric to aid in predicting the true number of states in the population.](./output/figure6.svg){#fig:6}

### Model performs reasonable in the presence of different censoring types in the data.

In this part we compare model performance in case where we have a full binary tree, and where the data in censored due to cell death and experiment end time. The upper row shows the lineage tree over time for full tree and censored data, the lower row represents state assignement accuracy for both cases.

![**Censored versus uncensored data. Model performes well even if we have censored data.](./output/figure7.svg){#fig:7}
