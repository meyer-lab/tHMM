# Results

## Simulated Lineage

To provide a better understanding of the accuracy and parameterization of the states by the tHMM, lineages of cells were simulated by starting with an initial cell and sampling the Bernoulli and exponential distributions for its respective fate and lifetime. If a cell divided, it then produced two new simulated cells that underwent the same process sampling from the same Bernoulli and exponential distributions. 

We simulate a heterogeneous lineage by constructing two subpopulations from different underlying distributions. An initial therapy-susceptible subpopulation was constructed with ${\theta_{B}}^{(1)}=0.8$, ${\lambda_{E}}^{(1)}=80$ (i.e. low replication rate and long lifetime). Moreover, a therapy-resistant cancerous subpopulation was constructed with ${\theta_{B}}^{(1)}=0.99$, ${\lambda_{E}}^{(1)}=20$ (i.e. high replication rate and short lifetime). These two lineages were then joined by connecting the cell in the last generation of the first lineage to the initial root cell of the second lineage as a parent-daughter chain. Because this simulated heterogeneous lineage tree contains one transition among the two underlying distributions, we call this our depth model of heterogeneity. The lineage model is analogous to a single cell proliferating over time to create a lineage, in which one cell undergoes a stochastic mutation to form a new subpopulation with different phenotypic properties.

This lineage was inputted into the tHMM, fitted with our adapted Baum-Welch algorithm, and the states of each cell were estimated using the adapted Viterbi algorithm utilizing parameters found from Baum-Welch fitting. The accuracy of the model was determined by the number of correct predictions by Viterbi, given that the true state of a cell was marked by the distribution it was derived from. The model predicted the correct state for a single lineage with 99.19% accuracy. The true simulated lineage is shown on the left panel of Figure 3, and the model prediction of the cell states are seen on the right panel. 

![** Model prediction accuracy a simulated heterogeneous lineage.** Nodes in blue represent therapy-susceptible cells with parameters of ${\theta_{B}}^{(1)}=0.8$, ${\lambda_{E}}^{(1)}=80$. Nodes in orange are therapy-resistant cancerous cells with parameters of ${\theta_{B}}^{(1)}=0.99$, ${\lambda_{E}}^{(1)}=20$. The edge color represents the lifetime of the cell it is connected to. Some cells only have one daughter because daughter cells that lived past an experimental end time were excised from the analysis. The true lineage (left) consists of an initially susceptible daughter cell which undergoes a mutation downstream that gives rise to a resistant subpopulation. The tHMM uses the end-of-life fate and lifetime of each of these cells to predict their state classification. The model’s perfomance (99.19% accurate) is shown on the right.. ...](./figures/figure3.svg){#fig:tfac}


## Lineage Length Scaling

Thus, the lineage was recreated 200 times with varying lengths, and the results of the accuracy prediction and distribution parameters $(\theta_{B}, \lambda_{E})$ are seen in Figure 4. The accuracy of the model in assigning states improves with increased number of cells within a lineage. Additionally, the parameter estimation approaches that of the true value for each subpopulation. In particular, the model is more accurate in assigning correct parameter estimates for resistant cells. This is attributed to robustness of the tHMM in identifying cell transitions and performing maximum likelihood estimation to generate an accurate $\bm{E}$ matrix (Eq(3)).

![** Model prediction accuracy on a simulated heterogeneous lineage (N=200).** Accuracy of tHMM state assignment (left), Bernoulli parameter estimation (middle), and exponential parameter estimation (right) for lineages composed of initially therapy-sensitive (blue) that then later transitioned to therapy-resistant (orange) subpopulations. Dotted horizontal lines at target accuracy and true population values are depicted. Solid line represents a 10-point moving average. ...](./figures/figure4.svg){#fig:tfac}

## Lineage Number Scaling

Although the tHMM accuracy improves as lineage length increases, obtaining large single lineages is dependent on the growth properties of the cells of interest. Thus, we experimented with lineages of length less than 10 in order to see if increasing number of lineages improve population predictions (Figure 5). This is analogous to increasing the number of initially seeded cells, which divide and form lineages.

![** Model prediction accuracy on a simulated cell population with varying numbers of lineages.** Accuracy of tHMM state assignment (left), Bernoulli parameter estimation (middle), and exponential parameter estimation (right) for populations of cells with different numbers of lineages. This is analogous to increasing the number of initial seeded cells. Lineages were composed of initially therapy sensitive (blue) that then later transitioned to resistant (orange) subpopulations. Dotted horizontal lines at target accuracy and true population values are depicted. Solid line represents a 10-point moving average. ...](./figures/figure5.svg){#fig:tfac}

## KL Divergence Scaling

## AIC Figure
