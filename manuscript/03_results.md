## Results

![**Workflow to obtain phenotypic measurements of cell fitness for use in the tHMM.** This procedure begins by using time-lapse images to trace parent-daughter linkages over an entire lineage. The lineage, possessing each cell-specific measure of lifetime and fate, is then used to model all latent states (i.e., cell subpopulations) in the sample using Bernoulli and exponential distributions that are fitted with Baum-Welch maximum likelihood estimation. After classifying each subpopulation, the pipeline assigns each cell from the lineage to its respective subpopulation (i.e., hidden state) which provides the quantitative measure for cell-specific resistance to cancer therapy. Created with BioRender.com.](./output/figure1.svg){#fig:workflow}

### Simulated Lineage

some introduction...

![**Distribution of alive cells in resistant and susceptible states over time and generation.** The distribution of alive cells in susceptible and resistant states after pruning the lineage, based on the parameterization including initial probability ${\pi = [0.6, 0.4]}$, transition probability matrix $[[0.85, 0.15],[0.15, 0.85]]$ with ${(p, shape, scale) = (0.99, 20, 5)}$ for resistant cells, and ${(p, shape, scale) = (0.88, 10, 1)}$ for susciptable cells. **a** and **c** show the number of cells over generations and time, and **b** and **d** show the proportion of those over generation and time, respectively.](./output/figure2.svg){#fig:Distributions}


To provide a better understanding of the accuracy and parameterization of the states by the tHMM, lineages of cells were simulated by starting with an initial cell and sampling the Bernoulli and exponential distributions for its respective fate and lifetime. If a cell divided, it then produced two new simulated cells that underwent the same process sampling from the same Bernoulli and exponential distributions. 

We simulate a heterogeneous lineage by constructing two subpopulations from different underlying distributions. An initial therapy-susceptible subpopulation was constructed with ${\theta_{B}}^{(1)}=0.8$, ${\lambda_{E}}^{(1)}=80$ (i.e. low replication rate and long lifetime). Moreover, a therapy-resistant cancerous subpopulation was constructed with ${\theta_{B}}^{(1)}=0.99$, ${\lambda_{E}}^{(1)}=20$ (i.e. high replication rate and short lifetime). These two lineages were then joined by connecting the cell in the last generation of the first lineage to the initial root cell of the second lineage as a parent-daughter chain. Because this simulated heterogeneous lineage tree contains one transition among the two underlying distributions, we call this our depth model of heterogeneity. The lineage model is analogous to a single cell proliferating over time to create a lineage, in which one cell undergoes a stochastic mutation to form a new subpopulation with different phenotypic properties.

This lineage was inputted into the tHMM, fitted with our adapted Baum-Welch algorithm, and the states of each cell were estimated using the adapted Viterbi algorithm utilizing parameters found from Baum-Welch fitting. The accuracy of the model was determined by the number of correct predictions by Viterbi, given that the true state of a cell was marked by the distribution it was derived from. The model predicted the correct state for a single lineage with 99.19% accuracy. The true simulated lineage is shown on the left panel of Figure 3, and the model prediction of the cell states are seen on the right panel.

### Heterogeneous model with 1 lineage per population

![Model prediction accuracy a simulated heterogeneous lineage. There is only one lineage per population. Nodes in blue represent therapy-susceptible cells with parameters of ${(p, shape, scale) = (0.88, 10, 1)}$. Nodes in orange are therapy-resistant cancerous cells with parameters of ${(p, shape, scale) = (0.99, 20, 5)}$. The upper row shows the estimation of the emission distributions, including bernoulli paramter, gamma shape, and gamma scale. Estimation accuracy of the Bernoulli parameter **a**, shape parameter of Gamma distribution **b**, and scale parameter of Gamma distribution **c**. The lower row exhibits the accuracy of model parameters, including state prediction of cells **d**, accuracy of initial probability estimation **e**, estimation accuracy of state transition probability matrix **f** for a range of cell numbers.](./output/figure3.svg){#fig:heterog1Model}


### Heterogeneous model with more than 1 lineage per population

![Model prediction accuracy a simulated heterogeneous lineage with more than 1 lineage per population. Nodes in blue represent therapy-susceptible cells with parameters of ${(p, shape, scale) = (0.88, 10, 1)}$. Nodes in orange are therapy-resistant cancerous cells with parameters of ${(p, shape, scale) = (0.99, 20, 5)}$. The upper row shows the estimation of the emission distributions, including bernoulli paramter, gamma shape, and gamma scale. Estimation accuracy of the Bernoulli parameter **a**, shape parameter of Gamma distribution **b**, and scale parameter of Gamma distribution **c**. The lower row exhibits the accuracy of model parameters, including state prediction of cells **d**, accuracy of initial probability estimation **e**, estimation accuracy of state transition probability matrix **f** for a range of cell numbers.](./output/figure4.svg){#fig:heteroMultiModel}


### Lineage Length Scaling

Thus, the lineage was recreated 200 times with varying lengths, and the results of the accuracy prediction and distribution parameters $(\theta_{B}, \lambda_{E})$ are seen in Figure 4. The accuracy of the model in assigning states improves with increased number of cells within a lineage. Additionally, the parameter estimation approaches that of the true value for each subpopulation. In particular, the model is more accurate in assigning correct parameter estimates for resistant cells. This is attributed to robustness of the tHMM in identifying cell transitions and performing maximum likelihood estimation to generate an accurate $\bm{E}$ matrix (Eq(3)).



### Lineage Number Scaling

Although the tHMM accuracy improves as lineage length increases, obtaining large single lineages is dependent on the growth properties of the cells of interest. Thus, we experimented with lineages of length less than 10 in order to see if increasing number of lineages improve population predictions (Figure 5). This is analogous to increasing the number of initially seeded cells, which divide and form lineages.


### Wasserstein Divergence Scaling

some introduction...
![Model prediction accuracy on a simulated heterogeneous lineage (N=200). Accuracy of tHMM state assignment (left), Bernoulli parameter estimation (middle), and exponential parameter estimation (right) for lineages composed of initially therapy-sensitive (blue) that then later transitioned to therapy-resistant (orange) subpopulations. Dotted horizontal lines at target accuracy and true population values are depicted. Solid line represents a 10-point moving average. ...](./output/figure6.svg){#fig:KL}


### AIC Figure

some introduction...
![Model prediction accuracy on a simulated cell population with varying numbers of lineages. Accuracy of tHMM state assignment (left), Bernoulli parameter estimation (middle), and exponential parameter estimation (right) for populations of cells with different numbers of lineages. This is analogous to increasing the number of initial seeded cells. Lineages were composed of initially therapy sensitive (blue) that then later transitioned to resistant (orange) subpopulations. Dotted horizontal lines at target accuracy and true population values are depicted. Solid line represents a 10-point moving average. ...](./output/figure7.svg){#fig:AIC}