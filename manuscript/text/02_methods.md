# Materials and Methods
Cell Culture Experiments
Cell Culture Set-up
To model a heterogeneous tumor, we conducted a proof-of-concept experiment by co-culturing cells derived from lung
adenocarcinoma: PC-9 and H1299. These cells were obtained from the Meyer Lab at UCLA, where PC-9 cells were already
transfected to fluoresce red through a H2B (histone h2b)-mKate2 nuclear localization lentivirus (Sigma-Aldrich, St. Louis,
MO) and H1299 green through a H2B-GFP lentivirus (Sigma-Aldrich, St. Louis, MO). Before co-culturing, PC-9 and H1299
cells were grown separately in monolayers in tissue culture-treated well plates, using complete media (Roswell Park
Memorial Institute media with L-glutamine (Thermo Fisher Scientific, Canoga Park, CA), supplemented with 5% fetal
bovine serum (Corning, Corning, NY) and 1% penicillin/streptomycin (Thermo Fisher Scientific, Canoga Park, CA)). Similar
conditions for the well plates and media were used for the co-culture experiment. Cells were passaged every 3 days and
thrown out after 9 passages.
Co-Culture Set-up and Imaging
To co-culture PC-9 and H1299 cells, 200 PC-9 cells/cm2
and 200 H1299 cells/cm2 were taken from their respective well
plates and seeded in complete media. Once seeded, cells were treated with 100 nM erlotinib (Selleckchem, Houston, TX),
an epidermal growth factor receptor inhibitor which causes significant growth inhibition in PC-9 cells (IC50 = 7 nM) but not
in H1299 cells (IC50 > 10 μM).34, 35 Cells were incubated in the Incucyte S3 Live-Cell Analysis System (Incucyte, Ann Arbor,
MI) and imaged using phase contrast and fluorescent microscopy. The green channel excitation and emission ranges were
440-480 nm and 504-544 nm, respectively, whereas the red channel excitation and emission ranges were 565-605 nm
and 625-705 nm, respectively. The exposure times were 300 ms for the green channel and 400 ms for the red channel. A
20x objective with a numerical aperture of 0.45 was used. Imaging started 24 hours after initial seeding to allow cells to
adhere, and images were acquired every 5 minutes for 4 days.
Cell Tracking and Lineage Generation
To generate lineage trees from the acquired images, a cell image analysis software, ilastik (European Molecular Biology
Laboratory, Heidelberg, Germany), was used to track the cells from the time-lapse datasets.36 Before inputting the image
sets into ilastik, Fiji’s ImageJ (NIH) Stackreg plugin was used to correct for in-plane drift.37 Moving to ilastik, pixel
classification was used to segment the images, and we employed a tracking with deep learning workflow to training the
program on true cell divisions and false detections. This allowed us to construct lineage tracks and link objects between
frames. Upon running the pipeline to completion, the output was exported as a comma-separated values (CSV) file
containing identification numbers for each cell and corresponding parent cells for each image. Using this CSV file, the
parameters of interest, such as cell fate, longevity, and cell type were extracted using the Python programming language.
Single state model
Notation and model description
We first build a model of cell growth based on phenotypic measurements of cells. The first measurement is the cell's fate,
encoded as 𝜙 where 𝜙 ∈ {0,1}, a binary outcome where 𝜙 = 0 is the cell dying at the end of its lifetime, and 𝜙 = 1 is the
cell dividing into two daughter cells. The second measurement is the cell's lifetime, encoded as 𝜏, where 𝜏 ∈ (0, +∞), a
positive real number indicating how long the cell lived in hours. For example, a complete observation could be of the form
𝒙𝒎 = (1,20) where cell 𝑚 divided into two daughter cells after living for 20 hours. In general, for any observation 𝒙𝒏 for
cell 𝑛, we have a tuple indicating the cell fate and the cell lifetime, 𝒙𝒏 = (𝜙𝑛, 𝜏𝑛
). To probabilistically model each
observation, the cell fate follows a Bernoulli distribution with Bernoulli rate parameter 𝑝𝐵 where 𝑝𝐵 ∈ [0,1] and 𝑝𝐵
represents the probability of 𝜙 = 1, the chance that a cell will divide. The cell lifetime follows a Gompertz distribution
with Gompertz rate parameter 𝑐𝐺 and scale parameter 𝑠𝐺. The Gompertz distribution models the mortality of cells over
time. These underlying parameters also describe the states in the multiple state model discussed later.
Integrative Biology 7
In the single state model, we assume that all the cells come from the same distribution. Using Maximum Likelihood
Estimation (MLE), we can fit all the cells and find the underlying set of parameters 𝑝𝐵, 𝑐𝐺, and 𝑠𝐺. We use the commonly
known and well-studied Bernoulli parameter estimator, where 𝑝̂𝐵 represents the estimate, defined as the following:
𝑝̂𝐵 =
1
𝑁
∑𝜙𝑛
𝑛
.
𝐸𝑞(1)
Summing up the observations over the total number of observations is the estimator of the mean or the expected value
for the Bernoulli distribution rate parameter.
As for the two Gompertz parameters, we estimated 𝑠̂𝐺 and 𝑐̂𝐺 through first finding a modified Gompertz parameter 𝑏̂ that
numerically minimized the following:
𝐿(𝑏) = |∑
𝜏𝑛𝑒
𝑏𝜏𝑛
∑ 𝑒
𝑏𝜏𝑛
𝑛
𝑁
𝑛 − 1
− ∑(
𝑒
𝑏𝜏𝑛 − 1
𝑏 ∑ 𝑒
𝑏𝜏𝑛
𝑖
𝑁
− 𝑏
+ 𝜏𝑛)
𝑛
|,
𝐸𝑞(2)
where 𝐿(𝑏) is the score function, the derivat
Integrative Biology 8
at the end of the lineage tree or cells at the leaves of the tree (nodes with only one edge) will be denoted by the set 𝑳. All
other cells (cells at nodes that are not leaves) will be denoted by the set 𝒏𝑳.
To fully describe both trees, we say that a joint distribution 𝑃(𝒁,𝑿) follows the tree hidden Markov property if and only if
𝑃(𝒁,𝑿) = 𝑃(𝒛𝟏, 𝒛𝟐, … , 𝒛𝑵, 𝒙𝟏, 𝒙𝟐, … , 𝒙𝑵) = 𝑃(𝒛𝟏
)∏𝑃( 𝒛𝒏 ∣
∣ 𝒛𝑷(𝒏) )∏𝑃( 𝒙𝒏 ∣ 𝒛𝒏
)
𝑁
𝑛=1
𝑁
𝑛=2
.
𝐸𝑞(4)
This factorization of the joint distribution follows from the conditional independence properties of our emissions (i.e.
observations) and the Markov tree dependence of the latent variables. These can be easily derived from the Bayesian
network diagram in Figure 1 which graphically shows the influence of each variable on the other. The similarity to the
factorization of hidden Markov chains (HMCs) is also evident, the main difference being that the transition probabilities
for tHMMs are 𝑃( 𝒛𝒏 ∣
∣ 𝒛𝑷(𝒏) ) and the transition probabilities for HMCs are 𝑃( 𝒛𝒏 ∣
∣ 𝒛(𝒏−𝟏) ).
Parameters
Each factor in the tree hidden Markov property represents a key parameter in the tHMM. Fully describing the tree Hidden
Markov property with known values specifies the entire model. The following parameters are similar to those used in
HMCs.
We first introduce the hyperparameter 𝐾 which is the number of possible discrete hidden states the hidden variables can
take. This is the only parameter the user is required to input as all the other parameters depend on the value of 𝐾.
Ultimately, each state 𝑘 ∈ {1,2, … ,𝐾} uniquely describes the distributions (Bernoulli and Gompertz distributions) via the
respective parameters (𝑝𝐵
(𝑘)
, 𝑐𝐺
(𝑘)
, and 𝑠𝐺
(𝑘)
) governing the respective observations or emissions (cell fate 𝜙 and cell
lifetime 𝜏) for a group of cells. By ascribing a group of cells in the lineage tree with a particular state 𝑘 ∈ {1,2, … ,𝐾}, we
can identify subpopulations of interest based on the ascribed states of other groups of cells. For example, if the root cell
was found to be of state 1, that is to say, 𝒛0 = 1, but all cells at the leaves further down in the lineage were found to be
of state 2, then it is reasonable to assume that sometime in the lineage, a transition between states 1 and 2 occurred.
One can then further interrogate and ascribe meaning to each of the states. That is, if state 1 described a Bernoulli
distribution with Bernoulli rate parameter 𝑝𝐵
(1) = 0.5 but state 2 described a Bernoulli distribution with Bernoulli rate
parameter 𝑝𝐵
(2) = 0.9, then state 2 can be identified as cells that are resilient or highly proliferative compared to cells of
state 1. The meaning ascribed to each state is furnished by the user upon interrogation of the distributions that each state
uniquely describes. Sometimes the number of states 𝐾 can be arbitrarily chosen; for example, if the number of states
selected equals the number of cells totally observed, that is, 𝐾 = 𝑁, then each cell will be ascribed its own unique state
and the goal of using our model to identify heterogeneity is trivialized. To prevent an arbitrary selection of 𝐾, the Akaike
information criterion (AIC) is used for model selection and can inform the user of what value of 𝐾 is best.39 AIC was
calculated using the following, where 𝐿𝐿 is negative of the log-likelihood.
𝐴𝐼𝐶 = 2(𝐿𝐿 + 𝐾(𝐾 − 1)).
𝐸𝑞(5)
Once the number of discrete states 𝐾 is chosen, the three other parameters describing tree hidden Markov property can
be built. The first parameter is a vector of initial hidden state priors or an initial probability distribution over the set of
states. This describes the probability 𝑃(𝒛𝟏
), or more explicitly, 𝑃(𝒛𝟏 = 𝑘) for some state 𝑘 ∈ {1,2, … ,𝐾}, which is then
encoded as 𝜋𝑘. That is to say, 𝜋𝑘 is the probability that the observed cell at the first hidden root node is of state 𝑘 for 𝑘 ∈
{1,2, … ,𝐾}. Note that for some states 𝑗 ∈ {1,2, … ,𝐾}, 𝜋𝑗 = 0 implying that they cannot be initial states. These initial
probabilities are stored as a 𝐾-dimensional vector 𝝅 where the 𝑗-th entry is 𝜋𝑗 = 𝑃(𝒛𝟏 = 𝑗) for 𝑗 ∈ {1,2, … ,𝐾}.
The second parameter is a matrix of state transition probabilities stored as a 𝐾 × 𝐾 matrix 𝑻. Each element of the matrix
𝑇𝑗𝑘 represents the probability of going to state 𝑘 (column-element) from state 𝑗 (row-element). More explicitly, 𝑇𝑗𝑘 =
Integrative Biology 9
𝑃( 𝒛𝒏 = 𝑘 ∣
∣ 𝒛𝑷(𝒏) = 𝑗 ) for some states 𝑗, 𝑘 ∈ {1,2, … ,𝐾}. Note that the diagonal of 𝑻 consists of entries of the form 𝑇𝑘𝑘,
the probability of staying in the same state as the parent cell.
The third parameter is 𝐾-long list of parameters for the distributions each state uniquely describes. We call this list 𝑬 =
{𝐸1,𝐸2, … , 𝐸𝐾} where for some 𝑘 ∈ {1,2, … ,𝐾}, the parameters describing the distributions of state 𝑘 are stored as 𝐸𝑘 =
{𝑝𝐵
(𝑘)
, 𝑐𝐺
(𝑘)
, 𝑠𝐺
(𝑘)
}. This parameter describes the final remaining factor in the tree hidden Markov property, the emission
likelihoods 𝑃( 𝒙𝒏 ∣ 𝒛𝒏
). More explicitly, for some state 𝑘 ∈ {1,2, … ,𝐾} and for some observation of cell fate and lifetime
(𝜙𝑛, 𝜏𝑛
), we are trying to describe the probability 𝑃( 𝒙𝒏 = (𝜙𝑛, 𝜏𝑛
) ∣ 𝒛𝒏 = 𝑘 ). Here we assume the observations are
conditionally independent given their respective state variable which is essentially a Naive Bayes assumption on the
emissions (each feature or observation (𝜙𝑛, 𝜏𝑛) is conditionally independent of the other given the category or state 𝑘):
𝑃( 𝒙𝒏 ∣ 𝒛𝒏
) = 𝑃( 𝒙𝒏 = (𝜙𝑛, 𝜏𝑛
) ∣ 𝒛𝒏 = 𝑘 ),
= 𝑃( 𝒙𝒏,𝟏 = 𝜙𝑛 ∣
∣ 𝒛𝒏 = 𝑘 ) × 𝑃( 𝒙𝒏,𝟐 = 𝜏𝑛 ∣
∣ 𝒛𝒏 = 𝑘 ),
= 𝑃 ( 𝒙𝒏,𝟏 = 𝜙𝑛 ∣
∣ 𝑝𝐵
(𝑘)
) × 𝑃 ( 𝒙𝒏,𝟐 = 𝜏𝑛 ∣
∣ 𝑐𝐺
(𝑘)
, 𝑠𝐺
(𝑘)
) ,
= 𝐸𝑘
(𝜙𝑛, 𝜏𝑛
).
𝐸𝑞(6)
We introduce the last line to avoid writing out the full product of the emission likelihoods every time it appears in an
expression. The parameters are reiterated in Table 1. Ultimately, using the parameters we can elaborate on the hidden
Markov tree property in Eq (4) as the following:
𝑃(𝒁,𝑿;𝝅, 𝑻, 𝑬) = 𝑃(𝒛𝟏, 𝒛𝟐, … , 𝒛𝑵, 𝒙𝟏, 𝒙𝟐, … , 𝒙𝑵;𝝅, 𝑻, 𝑬), = 𝑃(𝒛𝟏;𝝅)∏𝑃( 𝒛𝒏 ∣
∣ 𝒛𝑷(𝒏)
; 𝑻 )∏𝑃( 𝒙𝒏 ∣ 𝒛𝒏;𝑬)
𝑁
𝑛=1
𝑁
𝑛=2
.
𝐸𝑞(7)