# Discussion

## Simulated Lineage

The model failed to correctly classify only the cell that underwent the mutation prior to the onset of the resistant subpopulation. This sole incorrect classification demonstrates that the model may not be highly sensitive to immediate cell transitions, particularly if the phenotypic behaviors of the parent and daughter are relatively similar. However, the model proved efficacious both upstream and downstream of the transition, though may be less accurate on lineages with fewer cells. 

## Lineage Length Scaling

There are several lineages, however, on which the tHMM performed poorly, in particular the susceptible cells possessed high variance in parameter estimation. The deviations are also seen in the resistant cells, as some of the exponential estimates were orders of magnitude higher than the true value. The exponential estimator, similar to any other lifetime distribution, suffered from survivorship bias due to removal of unfinished cells that were still alive at the end of the tracking period and initial cells that failed to divide and create a lineage. Specifically, cells with longer lifetimes (i.e. higher growth rate) are more likely to live beyond the tracking period and thus are excluded from model estimation. This unavoidable phenomenon leads to the growth rate parameter approaching a value less than the true value (dotted line on figure) and biased yet precise estimation. Although Bernoulli estimations are centered around their respective true values, they suffer from survivorship bias as well because cells with higher θ_B divide more often and thus have a higher sample size for prediction. This leads to the resistant cell line possessing more accurate Bernoulli estimations relative to the susceptible subpopulation.

## Lineage Number Scaling

The tHMM accuracy performs maximum likelihood estimation using cell observations from each lineage in the population. Thus, the improved performance accuracy and decrease in its variance as lineage number increases validates the model architecture. Parameter estimation was poor on the onset, as the first two Bernoulli parameters for each state overlapped (as seen by the dark shaded circles), denoting that the model placed all cells into a single state. The separate subpopulations estimations were initially poor compared to true value, though the tHMM was able to better distinguish the separate subpopulations as more lineages were added.  
 
