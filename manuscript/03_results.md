## Results

### A figure to discribe it all
TODO at last.


### How to use tHMM model with different Phenotypes
Any observable heritable phenotype could be used as Emissions. The following (Figure 2A) shows schematic of a lineage tree in which we can observe phenotypes such as, cell fate, inter mitotic times (lifetime), G1 cell cycle phase duration, G2 cell cycle phase duration, and cell shape. For each of these features we consider a distribution, shown in parts b-e.

![**This figure shows the flexibility of the model and that we can use any tracktable phenotype.](./output/figure2.svg){#fig:2}


### How big the experiments need to be
Here we show the performance of the model by increasing the cell numbers. The figure shows the state assignment accuracy and parameter estimations.

![**A figure to show how many cells we would need to obtain a reasonable performance.](./output/figure3A.svg){#fig:3A}


### How far the states need to be
Employing Wasserestein distance, we show the more different the states, the higher the accuracy of state assignments.

![**Wasserestein distance shows in a 2-state population, the accuracy increases if the states are farther to each oteher.](./output/figure3B.svg){#fig:3B}


### What if we have rare phenotypes in the data
This parts explains how the model performs in the case of an under-represented population of cells.

![**State assignment accuracy when we have different proportions of each state, showing the model performs well even when there is an under- (or over-) represented population.](./output/figure4.svg){#fig:4}

### Determining the number of states
In this part we explain the AIC figures.

### Model performance in case of censored data
In this part we compare model performance in case where we have a full binary tree, and where the data in censored due to cell death and experiment end time. The upper row shows the lineage tree over time for full tree and censored data, the lower row represents state assignement accuracy for both cases.

![**Censored versus uncensored data. Model performes well even if we have censored data.](./output/figure6.svg){#fig:6}
