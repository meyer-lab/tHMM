# Materials and Methods

## Cell Culture

### Extracting Phenotypic Parameters
?
#### End-of-Life Fate and Lifetime
?
#### Cell Cycle Markers
?
### Cell Culture Set-up and Tracking for Lineage Generation
?
## Single state model

Our group first built a synthetic model of cell growth based on phenotypic measurements of cells. The first measurement is the cell's fate, encoded as $\phi$ ,where $\phi\in\{0,1\}$, a binary outcome where $\phi=0$ defines cell death at the end of its lifetime, and $\phi=1$ is the cell dividing into two daughter cells. The second measurement is the cell's lifetime, encoded as $\tau$, where $\tau\in (0, +\infty)$, a positive real number indicating how long the cell lived in hours. For example, a complete observation could be of the form $\bm{x}_{m} = (1,3)$ where cell $m$ divided into two daughter cells after living for 3 hours. In general, for any observation $\bm{x}_{n}$ for cell $n$, we have a tuple indicating the cell fate and the cell lifetime, $\bm{x}_{n}=(\phi_{n}, \tau_{n})$. The observations or emissions can be expanded and manipulated to fit any type of phenotypic observation, such as measurements from live-cell reporters. For example, to probabilistically model the observations mentioned earlier, the cell fate follows a Bernoulli distribution with Bernoulli rate parameter $\theta_{B}$ where $\theta_{B}\in[0,1]$ and $\theta_{B}$ represents the probability of $\phi=1$, the chance that a cell will divide. The cell lifetime follows an exponential distribution with growth rate parameter $\lambda_{E}$. The exponential distribution models the mortality of cells over time. These underlying parameters also describe the states in the multiple state model discussed later.

 
## Multiple state model

### Model description
 
Using the parent-daughter links of each cell, we can construct synthetic lineage trees which then capture information about the history of cells in the context of their families. Furthermore, for each observation $\bm{x}_{n}$ corresponding to cell $n$ in our lineage tree, we introduce a latent or hidden variable $\bm{z}_{n}$ which takes one of $K$ discrete values, $\bm{z}_{n}\in\{1,2,\ldots,K\}$. Ultimately, our proposed tHMM is comprised of two trees, an observed probabilistic tree, and a hidden probabilistic tree. The observed probabilistic tree is defined by the tree set $\bm{X}=\left\lbrace\bm{x}_{1},\bm{x}_{2},\ldots,\bm{x}_{N}\right\rbrace$ where $N$ is the number of total cells or observations in our lineage tree. Furthermore, we have a similar construction of the hidden probabilistic tree, defined by the tree set $\bm{Z}=\left\lbrace\bm{z}_{1},\bm{z}_{2},\ldots,\bm{z}_{N}\right\rbrace$. The hidden probabilistic tree and the observed probabilistic tree have the same indexing structure. It is important to note that the cell at node n is not necessarily the parent of the cell at node $n+1$, and that the cell at node $n$ is not necessarily the daughter of the cell at node $n-1$. However, because it is important to denote such relationships not only when describing lineage trees, but also when describing the relevant probability distributions, we introduce $\bm{P}(n)$ to denote the cell that is the parent of the cell at node $n$, and $\bm{C}(n)$ to denote the set of children of the cell at node $n$. Furthermore, the cell at node  n=1 or the root node will always be the initial cell in the lineage tree or the root cell. Cells at the end of the lineage tree, that is to say, cells at the leaves of the tree (nodes with only one edge or only one adjacent node) will be denoted by the set $\bm{L}$. All other cells (cells at nodes that are not leaves) will be denoted by the set $\bm{nL}$.

 
To fully describe both trees, we say that a joint distribution ${P}(\bm{Z},\bm{X})$ follows the tree hidden Markov property if and only if ${P}(\bm{Z},\bm{X})$ follows the tree hidden Markov property if and only if 

$${{P}(\bm{Z},\bm{X}) = P}(\bm{z}_{1},\bm{z}_{2},\ldots,\bm{z}_{N},\bm{x}_{1},\bm{x}_{2},\ldots,\bm{x}_{N}) = {P}(\bm{z}_{1})\prod_{n=2}^{N}{P}(\bm{z}_{n}\mid\bm{z}_{\bm{P}(n)})\prod_{n=1}^{N}{P}(\bm{x}_{n}\mid\bm{z}_{n})$$ 

This factorization of the joint distribution follows from the conditional independence properties of our emissions (i.e. observations) and the Markov tree dependence of the latent variables. These can be derived from the Bayesian network diagram in Figure 2 which graphically shows the influence of each variable on the other. The similarity to the factorization of hidden Markov chains (HMCs) is also evident, the main difference being that the transition probabilities for tHMMs are ${P}(\bm{z}_{n}\mid\bm{z}_{\bm{P}(n)})$ and the transition probabilities for HMCs are ${P}(\bm{z}_{n}\mid\bm{z}_{(n-1)})$. Each factor in the tree hidden Markov property represents a key parameter in the tHMM. Fully describing the tree Hidden Markov property with known values specifies the entire model. The following parameters are similar to those used in HMCs.

### Parameters

Each factor in the tree hidden Markov property in represents a key parameter in the tHMM. Fully describing it with known values specifies the entire model. The following parameters are similar to those used in describing HMCs.

We first introduce the hyperparameter $K$ which is the number of possible discrete hidden states the hidden variables can take. This is the only parameter the user is required to input, as all the other parameters depend on the value of $K$. Ultimately, each state $k\in\{1,2,\ldots,K\}$ uniquely describes the distributions (Bernoulli and exponential distributions) via the respective parameters (${\theta_{B}}^{(k)}$ and ${\lambda_{B}}^{(k)}$) governing the respective observations (i.e. the emissions of cell fate $\phi$ and cell lifetime $\tau$) for a group of cells. By ascribing a group of cells in the lineage tree with a particular state $k\in\{1,2,\ldots,K\}$, we can identify subpopulations of interest based on the ascribed states of other groups of cells. For example, if the root cell was found to be of state $1$, that is to say, $\bm{z}_{1}=1$, but all cells at the leaves further down in the lineage were found to be of state $2$, then it is reasonable to assume that in the lineage, a transition between states $1$ and $2$ occurred.

One can then further interrogate and ascribe meaning to each of the states. That is, if state $1$ described a Bernoulli distribution with Bernoulli rate parameter ${\theta_{B}}^{(1)}=0.5$ but state $2$ described a Bernoulli distribution with Bernoulli rate parameter ${\theta}^{(2)}=0.9$, then state 2 can be identified as cells that are resilient or highly proliferative compared to cells of state $1$. The meaning ascribed to each state is furnished by the user upon interrogation of the distributions that each state uniquely describes. Sometimes the number of states $K$ can be arbitrarily chosen; for example, if the number of states selected equals the number of cells totally observed, that is, $K=N$, then most cells will be ascribed their own unique state, and the goal of using our model in identifying heterogeneity is trivialized. To prevent an arbitrary selection of K, the Akaike information criterion (AIC) is used for model selection and can inform the user of what value of K is best.[@Robles] AIC is calculated using the following, where LL is negative of the log-likelihood.

$$AIC= 2(LL+K(K-1))$$
    
Once the number of discrete states $K$ is chosen, the three other parameters describing tree hidden Markov property can be built. The first parameter is a vector of initial hidden state priors or an initial probability distribution over the set of states. This describes the probability $P(\bm{z}_{1})$, or more explicitly, $P(\bm{z}_{1}=k)$ for some state $k\in\{1,2,\ldots,K\}$, which is then encoded as $\pi_{k}$. That is to say, $\pi_{k}$ is the probability that the observed cell at the first hidden root node is of state $k$ for $k\in\{1,2,\ldots,K\}$. Note that for some states $j\in\{1,2,\ldots,K\}$, $\pi_{j}=0$ implying that they cannot be initial states. These initial probabilities are stored as a $K$-dimensional vector $\bm{\pi}$ where the $j$-th entry is $\pi_{j}=P(\bm{z}_{1}=j)$ for $j\in\{1,2,\ldots,K\}$. This parameter can inform the user the initial chance a cell could be one of the latent states, such as identifying how likely a root cell in a lineage is likely to be resistant to therapy already.    
    
The second parameter is a matrix of state transition probabilities stored as a $K\times K$ matrix $\bm{T}$. Each element of the matrix $T_{jk}$ represents the probability of going to state $k$ (column-element) from state $j$ (row-element). More explicitly, $T_{jk} = {P}(\bm{z}_{n}=k\mid\bm{z}_{\bm{P}(n)}=j)$ for some states $j,k \in \{1,2,\ldots,K\}$. Note that the diagonal of $\bm{T}$ consists of entries of the form $T_{kk}$, the probability of staying in the same state as the parent cell. This parameter can inform the user the rate of transition between possible latent states, such as the probability of transitioning from a susceptible ($S$) cell to a resistant ($R$) one by looking at element $T_{SR}$.

The third parameter is $K$-long list of parameters for the distributions each state uniquely describes. We call this list $\bm{E}=\{E_{1},E_{2},\ldots,E_{K}\}$ where for some $k\in\{1,2,\ldots,K\}$, the parameters describing the distributions of state $k$ are stored as $E_{k}=\{{\theta_{B}}^{(k)},{\lambda_{E}}^{(k)}\}$. This parameter describes the final remaining factor in the tree hidden Markov property, the emission likelihoods ${P}(\bm{x}_{n}\mid\bm{z}_{n})$. More explicitly, for some state $k\in\{1,2,\ldots,K\}$ and for some observation of cell fate and lifetime $(\phi_{n}, \tau_{n})$, we are trying to describe the probability ${P}(\bm{x}_{n}=(\phi_{n}, \tau_{n})\mid\bm{z}_{n}=k)$. Here we assume the observations are conditionally independent given their respective state variable which is essentially a Naive Bayes assumption on the emissions (each feature or observation $\phi_{n},\tau_{n}$ is conditionally independent of the other given the category or state $k$):


This parameter can inform the user how likely each cell is going to be of a certain state. Ultimately, using the parameters we can elaborate on the hidden Markov tree property in Eq(1) as the following:

\begin{multline}
{P}(\bm{\bm{Z}},\bm{\bm{X}}; \bm{\pi},\bm{T},\bm{E}) = {P} (\bm{z}_{1},\bm{z}_{2},\ldots,\bm{z}_{N},\bm{x}_{1},\bm{x}_{2},\ldots,\bm{x}_{N};\bm{\pi},\bm{T},\bm{E}) \\ = {P}(\bm{z}_{1}; \bm{\pi})\prod_{n=2}^{N}{P}(\bm{z}_{n}\mid\bm{z}_{\bm{P}(n)}; \bm{T})\prod_{n=1}^{N}{P}(\bm{x}_{n}\mid\bm{z}_{n}; \bm{E})
\end{multline}
        
        
    
