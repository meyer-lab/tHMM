.. _installation:

.. highlight:: shell

============
Installation
============


From sources
------------

The sources for tHMM can be downloaded from the `Github repo`_.

You can clone the public repository:

.. code-block:: console

    $ git clone https://github.com/meyer-lab/tHMM.git

tHMM is built and its dependencies are managed with `uv`_. Once you have
`uv` installed, set up the environment and install all dependencies with:

.. code-block:: console

    $ cd tHMM
    $ uv sync

You can then run the test suite:

.. code-block:: console

    $ uv run pytest

or use any of the package's modules with:

.. code-block:: console

    $ uv run python -c "from lineage.LineageTree import LineageTree"

.. _Github repo: https://github.com/meyer-lab/tHMM
.. _uv: https://docs.astral.sh/uv/
