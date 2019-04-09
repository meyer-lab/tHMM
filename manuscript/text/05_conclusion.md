# Conclusion

The cell imaging protocol and tHMM pipeline construct and analyze cell lineage trees to properly assign cells to different
states based on cell fitness phenotypic properties. The current version of this pipeline is most accurate when populations
consist of at least 20 lineages with more than 600 cells per lineage; however, limitations in image resolution of available
microscopes may prevent accurate in vitro analysis. Upon utilizing longer experiment times and a wider field of view, the
tHMM pipeline may analyze heterogeneous single-cell growth for basic science research and resistance to chemotherapy.
The tHMM outputs cell fitness parameters – such as propensity to divide and expected lifetime – for each cell in its
respective subpopulation once the optimal number of states are found using AIC. The pipeline may provide researchers
and clinicians with improved classification of heterogeneity among cells, and provide information about unseen changes
in cellular identity.