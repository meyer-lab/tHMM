## Results

### A tree hidden Markov model can disentangle single-cell states in a population, given only heritable observations.

TODO at last.

### tHMM can be trained on several observable phenotypes as emissions.
Any observable heritable phenotype with a definable distribution could be used as an emission. The following (Figure 2) shows a schematic of a lineage tree in which we can observe phenotypes such as, cell fate, inter-mitotic times (lifetimes), G1 cell cycle phase duration, G2 cell cycle phase duration, and cell shape. For each of these features we consider a probabilistic distribution, shown in parts b-e.

![**This figure shows the flexibility of the model and that we can use any tracktable phenotype.](./output/figure2.svg){#fig:2}

### tHMM performs accurately on experimentally-reasonable large lineages.
Here we show the performance of the model by increasing the cell numbers. The figure shows the state assignment accuracy and parameter estimations for censored lineages.

![**A figure to show how many cells we would need to obtain a reasonable performance.](./output/figure3.svg){#fig:3}

### tHMM performs best when the difference between cell states is significant.
Determining a threshold of some quality or characteristic to define "different states" in the context of cell biology is complicated. One can consider a continuous spectrum of a phenotype within a heterogeneous tumor of cells. For example, in a population one can consider infinite states corresponding to the proliferation speed. Moreover, with the assumption of discrete and finite number of states, it is hard to draw a line and separate fast from slow proliferating cells, especially if the difference is slight. To simplify this concept, we consider the cell states to be discrete and finite. In either case, if the phenotype that we are tracing as an emission is more distinct between the states, clustering cells with that phenotype and state assignment associated with cells is easier.  
In order to test the performance of our model given different cases as the states being similar or different, we employed Wasserstein distance metric. In our setup, the emissions include cell fate (in G1 and S/G2), G1 phase duration, and S/G2 phase duration in two states. To represent the difference between states associated with these phenotypes, we kept cell fate and S/G2 duration parameters constant and varied a G1 phase duration parameter across the states. We sweep a space where at the beginning the two states are very similar and they becomre more and more different. The shared parameters associated with the emission between the two states are ($bern_{G_1} = 0.99$, $bern_{G_2} = 0.8$, $shape_{G_1} = 12$, $shape_{G_2} = 10$, $scale_{G_2} = 5$) and the scale parameter of G1 phase for state 1 is being varied in the range of $[1, 2.5]$, while the scale parameter of G1 in state 2 is kept at $1$.  
Figure 4a depicts the similarity and difference between two states in terms of cell shapes. The upper row represents state 1 with a gradual change of states over four stages, while the lower row shows state 2 which is kept the same so that the two states become more and more different over the four stages. Figure 4b presents the distribution of the cells' G1 duration for the two states and that first the two states have all their phenotypes identical, and G1 duration in state 1 changes so they become distinct. In Figure 4c, the accuracy of state assignment is shown where for the case where the two states are the same, Wasserstein distance is approximately zero and as the two states become more distinct the Wasserstein distance gets larger, meaning they are more different. Intuitively, if the states are more distinct, determining single-cells' states is more accurate. As shown in Figure 4c, the model is capable of determining the true states of single-cells when they are almost similar by almost $80%$ accuracy and reaches above $90$ when they are further distinct.

![Accuracy of single-cell state assignment given similar or different states present in a population. (a) A cartoon depicting an evolution in one state in a population given the other state stays the same, to elaborate on the performance of the model in similar or different existing states in a population. (b) The distribution of a phenotype which is varied in state 1 (blue) and the other state is kept constant (sienna). (c) The accuracy of state assignment versus the Wasserstein distance between the two states. It shows the accuracy increases if the states are more distant from each other.](./output/figure4.svg){#fig:4}

### tHMM can pick up even rare phenotypes within the lineage.

This parts explains how the model performs in the case of an under-represented population of cells.

![**State assignment accuracy when we have different proportions of each state, showing the model performs well even when there is an under- (or over-) represented population.](./output/figure5.svg){#fig:5}

### AIC metric indicates the model's capability to detemine the closest number of states in a population.

In this part we explain the AIC figures.

![**AIC of the model varying the number of states predicted, showing that AIC can be used as a metric to aid in predicting the true number of states in the population.](./output/figure6.svg){#fig:6}

### Model performs reasonable in the presence of different censoring types in the data.

In this part we compare model performance in case where we have a full binary tree, and where the data in censored due to cell death and experiment end time. The upper row shows the lineage tree over time for full tree and censored data, the lower row represents state assignement accuracy for both cases.

![**Censored versus uncensored data. Model performes well even if we have censored data.](./output/figure7.svg){#fig:7}
