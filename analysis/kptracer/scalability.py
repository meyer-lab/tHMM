"""Runtime of the E-step and of a full fit against tree size. Writes ``results/scalability.csv``."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import time
from pathlib import Path

import numpy as np
import pandas as pd
from simulation_study import PI_TRUE, Q_TRUE, D, K

from lineage.phylo_sim import simulate_dataset
from lineage.phyloHMM import PhyloHMM
from lineage.states.StateDistributionGaussian import StateDistribution as Gaussian
from lineage.tree_io import to_lineage

OUT = Path(__file__).parent / "results"

if __name__ == "__main__":
    rows = []
    rng = np.random.default_rng(0)
    means = rng.normal(0, 1.0, (K, D))
    for n_leaves in [100, 300, 1000, 3000, 10000, 30000]:
        data = simulate_dataset(1, n_leaves, Q_TRUE, PI_TRUE, means, 1.0, rng=rng)
        d = data[0]
        X = [to_lineage(d["tree"], d["tree"].leaf_names, d["obs"][d["tree"].is_leaf], [Gaussian(dim=D)])]
        m = PhyloHMM(X, K, rng=0)
        m.init_emissions()
        m.e_step()  # warm up
        t0 = time.perf_counter()
        reps = 3
        for _ in range(reps):
            post = m.e_step()
        t_e = (time.perf_counter() - t0) / reps
        t0 = time.perf_counter()
        m.m_step(post)
        t_m = time.perf_counter() - t0
        t0 = time.perf_counter()
        m2 = PhyloHMM(X, K, rng=0).fit(tol=1e-5)
        t_fit = time.perf_counter() - t0
        rows.append(
            {
                "n_leaves": n_leaves,
                "n_nodes": len(X[0]),
                "estep_s": t_e,
                "mstep_s": t_m,
                "fit_s": t_fit,
                "iters": len(m2.LL_trace),
            }
        )
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "scalability.csv", index=False)
