====
tHMM
====

``tHMM`` is a Python3 package for clustering, visualizing, and analyzing data in the form of lineage trees. This work is now published in Communications Biology https://www.nature.com/articles/s42003-022-04208-9


Install
-------

.. code-block:: console

    $ git clone https://github.com/meyer-lab/tHMM.git

It takes a few seconds to clone the repository.

Software Requirements
---------------------

This package is supported on *macOS* and *Linux* systems.

Python dependencies
~~~~~~~~~~~~~~~~~~~

Dependencies are managed with `uv <https://docs.astral.sh/uv/>`_.
Install the environment and dependencies by running:

.. code-block:: console

    $ uv sync

Quick start
-----------

Run the test suite, or use any of the package's modules directly:

.. code-block:: console

    $ uv run pytest
    $ uv run python -c "from lineage.LineageTree import LineageTree"

Citation
--------

If you use ``tHMM`` in your work, please cite:

    Mohammadi F, Visagan S, Gross SM, Karginov L, Lagarde JC, Heiser LM, Meyer AS.
    A lineage tree-based hidden Markov model quantifies cellular heterogeneity and plasticity.
    *Communications Biology* 5, 1258 (2022).
    https://doi.org/10.1038/s42003-022-04208-9

Citation metadata is also available in ``CITATION.cff``.

Documentation
-------------

Please find the documentation at https://meyer-lab.github.io/tHMM/.

License
-------

This project is covered under the MIT License.
