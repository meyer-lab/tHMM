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
	poetry run pytest -v -s -x

mypy:
	poetry run mypy --install-types --non-interactive --ignore-missing-imports lineage

testprofile:
	poetry run python3 -m cProfile -o profile -m pytest -s -v -x
	gprof2dot -f pstats --node-thres=5.0 profile | dot -Tsvg -o profile.svg

clean:
	git clean -fdx output
