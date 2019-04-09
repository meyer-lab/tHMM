# Results
Simulated Cells
To provide a better understanding of the accuracy and parameterization of the states by our tHMM prior to using it on
data from cultured cell lines, lineages of cells were simulated by starting with an initial cell, sampling the Gompertz
distribution for lifetime, then sampling the Bernoulli distribution for its fate. If a cell divided, it then produced two new
simulated cells that underwent the same process sampling from the same Gompertz and Bernoulli distributions (for
homogeneous lineages). This process was then terminated up to a given simulated experimental end time. If cells were
given a lifetime greater than the experimental end time, then their fates were not chosen and were labeled as unfinished.
Unfinished cells were then excised from the final lineage prior to analysis. Furthermore, initial root cells that were given a
death as their fate from the Bernoulli distribution were excised from the population since they formed single-cell lineages
and our tHMM required at least one transition within a lineage. Simulated cells were assumed to start their life the
moment their parent cell ended, and the two distributions were sampled independently; the value of the lifetime from
sampling the Gompertz distribution had no bearing on the value of the fate from sampling the Bernoulli distribution.
We simulated heterogeneous lineages in two ways. The first approach was to construct two lineages from different
underlying distributions. An initial lineage was constructed with one set of parameters (𝑝𝐵
(1)
, 𝑐𝐺
(1)
, and 𝑠𝐺
(1)
) up to some
experimental end time. Another lineage was constructed with another set of parameters (𝑝𝐵
(2)
, 𝑐𝐺
(2)
, and 𝑠𝐺
(2)
) up to
another experimental end time. These two lineages were then joined by connecting the cell in the last generation of the
first lineage to the initial root cell of the second lineage as a parent-daughter chain. We then ensure that the life start and
end times of the cells in the second lineage are offset based on the life end time of the cell we connected to in the initial
lineage. Because this simulated heterogeneous lineage tree only contains one transition between one possible underlying
distribution to another, we call this our depth model of heterogeneity.
The other model of heterogeneity we implemented was by constructing a lineage as previously described, but then
indicating a time point for transition. Cells would initially start out following the distributions dictated by the first given set
of parameters (𝑝𝐵
(1)
, 𝑐𝐺
(1)
, and 𝑠𝐺
(1)
) and then switch to follow the second set second parameters (𝑝𝐵
(2)
, 𝑐𝐺
(2)
, and 𝑠𝐺
(2)
) at a
distinct time point. This model allows for an unknown amount of transitions (because this would depend on how long
some cells live) but it does create a lineage wide transition specific up to a single generation (the generations of cells
before the given transition time point will be different from the generations after it). We call this second model our
breadth model of heterogeneity.
Model performance on simulated cells
An in silico lineage of two states was created using the depth model of heterogeneity. The initial distribution of cells
possessed the following set of parameters, (𝑝𝐵
(1)
= 0.99, 𝑐𝐺
(1)
= 2, and 𝑠𝐺
(1)
= 30), while the second cell distribution
possessed (𝑝𝐵
(2)
= 0.7, 𝑐𝐺
(2)
= 1.5, and 𝑠𝐺
(2)
= 25). This lineage was inputted into the tHMM, fitted with our adapted BaumWelch algorithm, and the states of each cell were estimated using the adapted Viterbi algorithm utilizing the parameters
found from Baum-Welch fitting. The accuracy of the model was determined by the number of correct predictions by
Viterbi, given that the true state of a cell is marked by the distribution it was derived from. The model predicted the
correct state for 85.1% of the cells in our depth model and for 95.3% of the cells in our breadth model (Figure 2). AIC was
performed on both a homogeneous lineage (one-state) and a depth heterogeneous (two-state) lineage and correctly
converged upon the proper number of states in both cases (Figure 3).
Cell Culture Experiments
Although a dose-response curve for each cell type and co-culture with erlotinib would prove useful in the future to
confirm appropriate drug concentration, we settled on 100 nM erlotinib which is much higher than erlotinib's IC50 of 7 nM
in PC-9 cells but lower than its IC 50 of over 10 μM in H1299 cells. From one experiment repeat of adding 100 nM erlotinib
to the co-culture experiment, PC-9 cells experienced longer division times as the assay progressed, whereas H1299 cell
division times remained stable (Figure 4). Furthermore, the number of imaged PC-9 cells increased from 25 cells at the 
Integrative Biology 11
start of imaging to 133 cells after 96 hours, whereas the number of imaged H1299 cells increased from 6 to 103 cells in
the same timeframe, verifying that H1299 cells grew faster than PC-9 cells in the presence of erlotinib. Lastly, there were
an average of 25 PC-9 cells and 63 H1299 cells per lineage over 96 hours.
Impact of population and lineage size on model performance
Upon varying the number of cells within lineages simulated according to our depth model of heterogeneity, the accuracy
of tHMM classification increased with an apparent logarithmic growth (Figure 5). Peak accuracy of >90% was observed
among lineages above ~600 cells. Estimation of the Bernoulli parameter for state 1 became closer to the true value as
lineage length increased, displaying a logarithmic trend similar to accuracy. Estimation of the Bernoulli parameter for
state 2 showed less accuracy and precision as lineage length increased. Estimation of Gompertz parameters, 𝑐𝐺 and 𝑠𝐺 for
both states was far less accurate than estimation of the Bernoulli parameter, demonstrated by high variance of points
from the true values.
The variation in number of lineages per population demonstrated no increase in state assignment accuracy (Figure 6).
However, there were substantial decreases in variance as lineage number increased. The Bernoulli parameter, on the
other hand, demonstrated relatively little variance in its estimation for both states. The Gompertz parameters were
noticeably high at lineage numbers of 8 and 9, and otherwise possessed no observable pattern.