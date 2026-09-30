# Latent-state tHMM on KP-Tracer phylogenies (issue #1014)

The findings are in [`REPORT.md`](REPORT.md). This directory holds the scripts that produce them.

## Data

Download the processed KP-Tracer release (Yang et al. 2022, Zenodo record 5847462, CC-BY, 1.3 GB):

```sh
mkdir -p ~/data/kptracer && cd ~/data/kptracer
curl -L -o KPTracer-Data.tar.gz https://zenodo.org/api/records/5847462/files/KPTracer-Data.tar.gz/content
tar xzf KPTracer-Data.tar.gz
```

The scripts look for `~/data/kptracer/KPTracer-Data` or `$KPTRACER_DATA`.

## Pipeline

Run from the repository root. `anndata` is only needed to read the h5ad, and `matplotlib` only for figures.

```sh
uv run python analysis/kptracer/simulation_study.py                 # results/simulation.csv
uv run python analysis/kptracer/scalability.py                      # results/scalability.csv (run from this dir)
uv run --with anndata python analysis/kptracer/prepare_data.py      # results/kptracer_inputs.pkl
cd analysis/kptracer
uv run python kptracer_analysis.py select        # K and branch-length selection  (~45 min on 46 cores)
uv run python kptracer_analysis.py fit 12        # anchored (fixed emissions), unsupervised K=12, sensitivity fits
uv run python kptracer_analysis.py bootstrap anchored
uv run python kptracer_analysis.py compare
uv run --with matplotlib python make_figures.py
uv run python bdmm_comparison.py                 # tHMM side of the BDMM-Prime comparison (bdmm/ holds the BEAST XML)
```

`results/*.pkl` are large intermediates and are git-ignored. The CSV summaries and `figures/` are committed.
