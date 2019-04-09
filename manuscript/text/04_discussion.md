# Discussion

The estimation of the Bernoulli parameter (𝑝𝐵) was much more accurate than the estimation of the two Gompertz
(𝑐𝐺, 𝑠𝐺) parameters (Figures 5, 6). The accuracy of the Bernoulli parameter estimation not only exceeded the accuracy of
the two Gompertz parameter estimations in the homogeneous single state mode, but in both heterogeneous models as
well. This finding was consistent among all simulated experiments due to use of an analytical estimator to obtain 𝑝𝐵 (Eq
(1)), as opposed to numerical for 𝑠𝐺 (Eq (2)). 𝑐𝐺 was then derived analytically from 𝑠𝐺(Eq (3)). One problem with the
Bernoulli estimator that was not present in the Gompertz estimation was survivorship bias. Because simulated cells with
high Bernoulli parameters could create more cells or samples from that distribution, estimating distributions with high
Bernoulli parameters was more accurate than estimating those with lower Bernoulli parameters.
The tHMM was capable of assigning cells to their correct state when simulated two-state lineages were analyzed with
over 600 cells (Figure 5). We anticipated accuracy to be the metric that increased as lineage length grew because
increasing the length of a lineage allowed for the effective sample size to increase. The tHMM, much like other
classification algorithms, was able to discriminate between distinct distributions with greater confidence when larger
sample sizes were given, leading to greater classification accuracy. Conversely, we did not see a significant increase in
parameter estimation accuracy when lineage number increased since our model operates on each lineage individually and
this set of simulated data only consisted of one lineage. When we increased the number of lineages in a population we
did not see a significant increase in state assignment accuracy (Figure 6). This result made intuitive sense because the
tHMM operated on each lineage independently and thus the accuracy should be roughly identical if all the lineages were
of the same length.
The tHMM correctly converged upon classifying all cells in the populations into two states upon performing an AIC (Figure
3). We believe that using AIC to determine the proper number of states will be a valuable tool for researchers and
clinicians who plan on using our analysis pipeline because the heterogeneous landscape of cancer cell fitness in response
to drug is rarely pre-determined. If not for AIC, our model would be classifying the cells into an arbitrarily selected number
of states – raising the problem in which the user could unnecessarily introduce heterogeneity into a model.
Preliminary experimental results demonstrated that, as expected, erlotinib inhibits PC-9 cell growth over time while not
affecting H1299 cell growth. In addition, these experiments provided information about the number and length of
lineages that the current protocol produces. In one well, 25 lineages were obtained with each lineage being 25 cells long
for PC-9, and 6 lineages of 63 cells for H1299. Comparing these quantities to model performance on simulated cells
indicated that more extensive data needs to be collected for the model to accurately classify cellular experimental data.
Nevertheless, the proof-of-concept experiment demonstrated heterogeneity within the cell population, as growth rates
were inhibited within each PC-9 cell lineage but unaffected within each H1299 cell lineage.
Our microscope currently divides each well into 16 fields of view at 20x, which makes it difficult to detect cells moving out
of a single image frame. Thus, these cells were excluded from their lineages and their measurements were not analyzed
by the tHMM. Moreover, the maximum measurable lineage length was limited by the length of the experiment. After 96
hours, the cell culture media was depleted by becoming too acidic from cell waste, preventing further cell growth.
Changing the media and re-adding erlotinib every 2-3 days may allow for longer experiment times. However, this would
require removal of the well plate from the Incucyte System, and re-insertion of the well plate may not guarantee that the
imaging positions remain the same.
One major input to our model is cell fate, which includes both cell division and cell death. However, from our co-culture
experiments, erlotinib did not cause cell death, which necessitates combining it with chemotherapeutic drugs to induce
death and better model heterogeneity and acquired resistance. One final obstacle is the behavior of cells in media
compared to the tumor microenvironment. Due to cell-cell interactions, mechanical stressors, and other factors, it is not
currently feasible to recreate the tumor microenvironment in vitro. As a result, creating a live tissue culture assay with an
imaging protocol analogous to frozen section analysis would be a more practical application of our cell tracking and tHMM
pipeline.40
