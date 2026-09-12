"""Test-session setup.

Several tests drive scipy/numpy linear algebra hard enough that BLAS's own thread
pool competes with pytest-xdist's worker processes for cores, which slows the suite
down rather than speeding it up. Pin each library to a single BLAS thread so the
parallelism comes from xdist alone; this must happen before numpy/scipy are
imported anywhere, so it lives at the top of the first conftest.py loaded.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
