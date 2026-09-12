.. _System-reqs:

.. highlight:: shell

===================
System Requirements
===================

Hardware requirements
=====================

`tHMM` package requires only a standard computer with enough RAM to support the in-memory operations.

Software requirements
=====================

OS requirements
---------------
    This package is supported for *macOS* and *Linux*. The package has been tested on the following systems:
    - macOS: Mojave (10.14.1)
    - Linux: Ubuntu 20.04

Python dependencies
-------------------
    `tHMM` requires Python >=3.10 and is built and managed with `uv <https://docs.astral.sh/uv/>`_.
    Once `uv` is installed, all required packages can be installed into a project-local
    virtual environment with ``uv sync``, run from the root of the repository.
    The package's dependencies are listed in ``pyproject.toml``.
