SHELL := /bin/bash

.PHONY: clean test testprofile testcover docs
flist = $(wildcard lineage/figures/figure*.py)

all: $(patsubst lineage/figures/figure%.py, output/figure%.svg, $(flist))

output/figure%.svg: lineage/figures/figure%.py
	if test -r "$@"; then \
		touch $@; \
	else \
		poetry run fbuild $*; \
	fi

test:
	poetry run pytest -s -v -x

mypy:
	poetry run mypy --install-types --non-interactive --ignore-missing-imports lineage

testprofile:
	poetry run python3 -m cProfile -o profile -m pytest -s -v -x
	gprof2dot -f pstats --node-thres=5.0 profile | dot -Tsvg -o profile.svg

clean:
	git clean -fdx output

# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)